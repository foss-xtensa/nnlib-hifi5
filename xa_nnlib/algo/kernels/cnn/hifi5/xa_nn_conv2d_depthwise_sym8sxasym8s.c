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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include <string.h>

#ifdef AE_MULZB3X3O8X8
#define DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
#endif

#if TFLITE_SINGLE_ROUNDING
#define MPY_BY_QUANT_MULT_X2_OUT32_NO_LR_SHIFT(inp, multiplier, shift) \
  MPY_BY_QUANT_MULT_X2_OUT32(inp, inp, multiplier, shift, shift);

#define MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(out, inp, multiplier, shift) \
  MPY_BY_QUANT_MULT_X2_OUT16(out, inp, multiplier, shift, shift);
#else
#define MPY_BY_QUANT_MULT_X2_OUT32_NO_LR_SHIFT(inp, multiplier, shift) \
  MPY_BY_QUANT_MULT_X2_OUT32(inp, inp, multiplier, XT_MAX(0, shift), XT_MAX(0, -shift));

#define MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(out, inp, multiplier, shift) \
  MPY_BY_QUANT_MULT_X2_OUT16(out, inp, multiplier, XT_MAX(0, shift), XT_MAX(0, -shift));
#endif

#define INTERLEAVE_3(dst0, dst1, src0, src1, src2) \
 { \
   ae_int8x8 tmp01_0; \
   tmp01_0 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src0), AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e060c04, 0x0a020800)));\
   AE_DSEL8X8(dst0, dst1, tmp01_0, AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea6200, 0xd9c84000)));\
 }

#define INTERLEAVE_4(dst0, dst1, src0, src1, src2, src3) \
 { \
   ae_int8x8 tmp01_0, tmp23_0; \
   tmp01_0 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src0), AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e060c04, 0x0a020800)));\
   tmp23_0 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT16X4(src3), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e060c04, 0x0a020800)));\
   AE_DSEL8X8(dst0, dst1, tmp01_0, tmp23_0, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea7362, 0xd9c85140)));\
 }


#define PACK_32X2(dst1, src1, src2) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

#define PACK_32X2_NEW(dst1, src1, src2, pattern) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), pattern);

#define DSEL32X4_HHLL(dst0, dst1, src0, src1) \
      {\
        ae_int8x8 temp0, temp1; \
        AE_DSEL8X8(temp0, temp1, AE_MOVINT8X8_FROMINT32X2(src0), AE_MOVINT8X8_FROMINT32X2(src1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbead9c8, 0x73625140))); \
        dst0 = AE_MOVINT32X2_FROMINT8X8(temp0); \
        dst1 = AE_MOVINT32X2_FROMINT8X8(temp1); \
      }

#ifndef AE_MULAZB8Q8X8CNV_H
#define MUL_CNV8Q8X8_H   AE_MULA8Q8X8CNV_H
#define MUL_CNV8Q8X8_L   AE_MULA8Q8X8CNV_L
#define MUL_CNV2X4Q8X8_H AE_MULA2X4Q8X8CNV_H
#define MUL_CNV2X4Q8X8_L AE_MULA2X4Q8X8CNV_L
#define MUL_CNV4O8X8_H   AE_MULA4O8X8CNV_H
#define MUL_CNV4O8X8_L   AE_MULA4O8X8CNV_L
#else
#define MUL_CNV8Q8X8_H   AE_MULAZB8Q8X8CNV_H 
#define MUL_CNV8Q8X8_L   AE_MULAZB8Q8X8CNV_L
#define MUL_CNV2X4Q8X8_H AE_MULAZB2X4Q8X8CNV_H
#define MUL_CNV2X4Q8X8_L AE_MULAZB2X4Q8X8CNV_L
#define MUL_CNV4O8X8_H   AE_MULAZB4O8X8CNV_H
#define MUL_CNV4O8X8_L   AE_MULAZB4O8X8CNV_L
#endif

/* 2D Convolution implementation */
static inline void conv2d_nchw_sym8sxasym8s_hf5_convmul
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_ker
  ,const WORD8 *__restrict__ p_inp
  ,WORD32 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height
  ,int actual_out_width
  ,int out_stride
  ,int x_stride
  ,int y_stride
  ,WORD32  input_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
  ,WORD32  out_zero_bias
  ,pWORD32 __restrict__ p_scratch
  )
{
  /* Importance of actual_out_width, since we are appending zeros input left
   * and right side. No problem with left padding, but for right padding that
   * is done to make sure that input_width is multiple of 4. Here
   * 'output_width_for_x_stride_1' value is calculated based on this padded
   * value. But actually expected output width to pick correct values from
   * 'output_width_for_x_stride_1' on jumps of 'x_stride'. */

  int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
  int kernel_width_pad = (kernel_width+3)&(~3);

  /* Generic case */
  int i, j, k, l;
  int output_width_for_x_stride_1;

  /* Here input_width is nothing but circ_buf_width, which is taken care to be
   * multiple of 4. */
  output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
  /* output_width_for_x_stride_1 loop is unrolled by 4 */
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, 8);

  ae_int32x2 d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3;
  ae_int32 *scratch_ptr = (ae_int32 *)p_scratch;

  ae_int32x2 d_bias;
  d_bias = AE_MOVDA32X2(bias, bias);

#ifndef AE_MULAZB8Q8X8CNV_H
  ae_int8x8 *pae_ker = (ae_int8x8 *)p_ker;
  ae_int8x8 d_izb = AE_MOVDA8(-input_zero_bias);
  ae_int64 d_acc64_0, d_acc64_1;
  d_acc64_0 = AE_ZERO64();
  d_acc64_1 = AE_ZERO64();
  ae_int32x2 d_sum_izbXker;
#else
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-input_zero_bias, 0));
  AE_MOVZBVCDR(biasvc1);
  ae_int32x2 zero32x2 = AE_ZERO32();
#endif

#ifndef AE_MULAZB8Q8X8CNV_H
#pragma loop_count min=1
  for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 3); i++)
  {
    AE_MULAAAA2Q8(d_acc64_0, d_acc64_1, d_izb, pae_ker[i]);
  }
  d_acc64_0 = AE_ADD64S(d_acc64_0, d_acc64_1);
  d_sum_izbXker = AE_SAT32X2(d_acc64_0, d_acc64_0);
#endif

  if(kernel_width_pad==12)
  {
    ae_int8x8 d_inp00, d_inp01, d_inp02;
    ae_int8x8 d_inp10, d_inp11, d_inp12;
    ae_int8x8 d_ker0, d_ker1, d_ker2;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j+=8)
      {
#ifndef AE_MULAZB8Q8X8CNV_H
        d_acc32_0 = AE_NEG32S(d_sum_izbXker);
        d_acc32_1 = AE_NEG32S(d_sum_izbXker);
        d_acc32_2 = AE_NEG32S(d_sum_izbXker);
        d_acc32_3 = AE_NEG32S(d_sum_izbXker);
#else
        d_acc32_0 = zero32x2;
        d_acc32_1 = zero32x2;
        d_acc32_2 = zero32x2;
        d_acc32_3 = zero32x2;
#endif
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
        ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + 1) * input_width) + j));
        ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
        for(k = 0; k < (kernel_height_pad>>1); k++)
        {
          AE_L8X8_IP(d_ker0, pt_ker, 8);
          AE_L8X8_IP(d_ker1, pt_ker, 8);
          AE_L8X8_IP(d_ker2, pt_ker, 8);
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, 8);
          AE_L8X8_XC(d_inp02, pt_inp0, (2*input_width-16));
          AE_L8X8_XC(d_inp10, pt_inp1, 8);
          AE_L8X8_XC(d_inp11, pt_inp1, 8);
          AE_L8X8_XC(d_inp12, pt_inp1, (2*input_width-16));

          MUL_CNV8Q8X8_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          MUL_CNV8Q8X8_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          MUL_CNV2X4Q8X8_H(d_acc32_0, d_acc32_1, d_ker1, d_inp01, d_inp10);
          MUL_CNV2X4Q8X8_L(d_acc32_2, d_acc32_3, d_ker1, d_inp01, d_inp02, d_inp10, d_inp11);
          MUL_CNV8Q8X8_L(d_acc32_0, d_acc32_1, d_ker2, d_inp10, d_inp11);
          MUL_CNV8Q8X8_H(d_acc32_2, d_acc32_3, d_ker2, d_inp11, d_inp12);
        }
        ae_int32x4 *p_sc = (ae_int32x4 *)(scratch_ptr + j);
        AE_S32X2X2_I(d_acc32_0, d_acc32_1, p_sc,  0);
        AE_S32X2X2_I(d_acc32_2, d_acc32_3, p_sc, 16);
      }
    }
  }
  else if(kernel_width_pad==8)
  {
    /* Regression is passing, but runperf.sh case is not matching with ref output,
    it is most probably due to kernel not being properly padded, need to fix this
    in testbench */
    ae_int8x8 d_inp00, d_inp01;
    ae_int8x8 d_inp10, d_inp11;
    ae_int8x8 d_ker0;
    ae_int8x8 d_ker1;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j+=8)
      {
#ifndef AE_MULAZB8Q8X8CNV_H
        d_acc32_0 = AE_NEG32S(d_sum_izbXker);
        d_acc32_1 = AE_NEG32S(d_sum_izbXker);
        d_acc32_2 = AE_NEG32S(d_sum_izbXker);
        d_acc32_3 = AE_NEG32S(d_sum_izbXker);
#else
        d_acc32_0 = zero32x2;
        d_acc32_1 = zero32x2;
        d_acc32_2 = zero32x2;
        d_acc32_3 = zero32x2;
#endif
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
        ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
        for(k = 0; k < (kernel_height_pad >> 1); k++)
        {
          AE_L8X8_IP(d_ker0, pt_ker, 8);
          AE_L8X8_IP(d_ker1, pt_ker, 8);
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, (input_width-8));
          AE_L8X8_XC(d_inp10, pt_inp0, 8);
          AE_L8X8_XC(d_inp11, pt_inp0, (input_width-8));

          MUL_CNV8Q8X8_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          MUL_CNV8Q8X8_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          MUL_CNV8Q8X8_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
          MUL_CNV8Q8X8_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
        }
        ae_int32x4 *p_sc = (ae_int32x4 *)(scratch_ptr + j);
        AE_S32X2X2_I(d_acc32_0, d_acc32_1, p_sc,  0);
        AE_S32X2X2_I(d_acc32_2, d_acc32_3, p_sc, 16);
      }
    }
  }
  else if(kernel_width_pad == 4)
  {
    ae_int8x8 d_inp00, d_inp01, d_inp02;
    ae_int8x8 d_inp10, d_inp11, d_inp12;
    ae_int8x8 d_ker0;
    ae_int32x2 d_acc32_4, d_acc32_5, d_acc32_6, d_acc32_7;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j+=16)
      {
#ifndef AE_MULAZB8Q8X8CNV_H
        d_acc32_0 = AE_NEG32S(d_sum_izbXker);
        d_acc32_1 = AE_NEG32S(d_sum_izbXker);
        d_acc32_2 = AE_NEG32S(d_sum_izbXker);
        d_acc32_3 = AE_NEG32S(d_sum_izbXker);
        d_acc32_4 = AE_NEG32S(d_sum_izbXker);
        d_acc32_5 = AE_NEG32S(d_sum_izbXker);
        d_acc32_6 = AE_NEG32S(d_sum_izbXker);
        d_acc32_7 = AE_NEG32S(d_sum_izbXker);
#else
        d_acc32_0 = zero32x2;
        d_acc32_1 = zero32x2;
        d_acc32_2 = zero32x2;
        d_acc32_3 = zero32x2;
        d_acc32_4 = zero32x2;
        d_acc32_5 = zero32x2;
        d_acc32_6 = zero32x2;
        d_acc32_7 = zero32x2;
#endif
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
        ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * (((i * y_stride + 0) * input_width) + j));
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + 0 + 1) * input_width) + j));

        ae_int8x8 *pt_ker0 = (ae_int8x8 *)(p_ker);

#pragma loop_count min=1
#pragma no_unroll
        for(k = 0; k < (kernel_height_pad >> 1); k++)
        {
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, 8);
          AE_L8X8_XC(d_inp02, pt_inp0, (2*input_width - 16));
          AE_L8X8_XC(d_inp10, pt_inp1, 8);
          AE_L8X8_XC(d_inp11, pt_inp1, 8);
          AE_L8X8_XC(d_inp12, pt_inp1, (2*input_width - 16));
          AE_L8X8_IP(d_ker0, pt_ker0, 8);

          MUL_CNV2X4Q8X8_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp10);
          MUL_CNV2X4Q8X8_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01, d_inp10, d_inp11);
          MUL_CNV2X4Q8X8_H(d_acc32_4, d_acc32_5, d_ker0, d_inp01, d_inp11);
          MUL_CNV2X4Q8X8_L(d_acc32_6, d_acc32_7, d_ker0, d_inp01, d_inp02, d_inp11, d_inp12);
        }
        ae_int32x4 *p_sc = (ae_int32x4 *)(scratch_ptr + j);
        AE_S32X2X2_I(d_acc32_0, d_acc32_1, p_sc,  0);
        AE_S32X2X2_I(d_acc32_2, d_acc32_3, p_sc, 16);
        if(output_width_for_x_stride_1 - j > 8)
        {
          AE_S32X2X2_I(d_acc32_4, d_acc32_5, p_sc, 32);
          AE_S32X2X2_I(d_acc32_6, d_acc32_7, p_sc, 48);
        }
      }
    }
  }
  else
  {
    ae_int8x8 d_inp00, d_inp01;
    ae_int8x8 d_inp10, d_inp11;
    ae_int8x8 d_ker0, d_ker1;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j+=8)
      {
#ifndef AE_MULAZB8Q8X8CNV_H
        d_acc32_0 = AE_NEG32S(d_sum_izbXker);
        d_acc32_1 = AE_NEG32S(d_sum_izbXker);
        d_acc32_2 = AE_NEG32S(d_sum_izbXker);
        d_acc32_3 = AE_NEG32S(d_sum_izbXker);
#else
        d_acc32_0 = zero32x2;
        d_acc32_1 = zero32x2;
        d_acc32_2 = zero32x2;
        d_acc32_3 = zero32x2;
#endif
        for(k = 0; k < kernel_height; k+=2)
        {
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j));
          ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + k + 1) * input_width) + j));
          /* Start of scratch memory for padded kernel is 8-byte aligned and
           * and kernel width is padded to be multiple of 4, so every alternate
           * row is 8-byte aligned for kernel */
          ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[k * kernel_width_pad]);
          ae_int8x8 *pt_ker1 = (ae_int8x8 *)(&p_ker[(k + 1) * kernel_width_pad]);
          ae_valign ker_a = AE_LA64_PP(pt_ker1);
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp10, pt_inp1, 8);
#pragma loop_count min=1
#pragma no_unroll
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_L8X8_XC(d_inp01, pt_inp0, 8);
            AE_L8X8_XC(d_inp11, pt_inp1, 8);
            AE_L8X8_IP(d_ker0, pt_ker0, 8);
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker1);
            MUL_CNV8Q8X8_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
            MUL_CNV8Q8X8_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
            MUL_CNV8Q8X8_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
            MUL_CNV8Q8X8_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
            d_inp00 = d_inp01;
            d_inp10 = d_inp11;
          }
        }
        if(kernel_width_pad & 4)
        {
#pragma loop_count min=1
#pragma no_unroll
          for(k = 0; k < kernel_height; k+=2)
          {
            ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j + (kernel_width_pad & (~7))));
            ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + k + 1) * input_width) + j + (kernel_width_pad & (~7))));
            ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[(k + 1) * kernel_width_pad - 4]);
            ae_int8x8 *pt_ker1 = (ae_int8x8 *)(&p_ker[(k + 2) * kernel_width_pad - 8]);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp10, pt_inp1, 8);
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_L8X8_XC(d_inp11, pt_inp1, 8);
              AE_L8X8_IP(d_ker0, pt_ker0, 8);
              AE_L8X8_IP(d_ker1, pt_ker1, 8);
              /* Lower half of d_ker0, d_ker1 is invalid here, it is ignored */
              MUL_CNV4O8X8_H(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
              MUL_CNV4O8X8_L(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
            }
          }
        }
        ae_int32x4 *p_sc = (ae_int32x4 *)(scratch_ptr + j);
        AE_S32X2X2_I(d_acc32_0, d_acc32_1, p_sc,  0);
        AE_S32X2X2_I(d_acc32_2, d_acc32_3, p_sc, 16);
      }
    }
  }

  /* Here we store output based on strides. For values in a row, values
   * will be picked from it as per 'x_stride'. No need to worry about
   * height dimension, since we took care of it by efficient row
   * accesses. */
  ae_int32 *scratch_ptr1 = (ae_int32 *) p_scratch;
#if TFLITE_SINGLE_ROUNDING
  int left_shift = out_shift;
  int right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift = XT_MAX(0, out_shift);
  int right_shift = XT_MAX(0, -out_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */

  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr1 = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
    WORD8 *out_ptr  = (WORD8 *) p_out + (i * out_stride * actual_out_width);
    ae_int32x2 accu_int32_0;
    ae_int8x8 accu_int8x8;

    for(j = 0; j < actual_out_width-1; j+=2)
    {
      accu_int32_0 = AE_SEL32_LL(scratch_ptr1[(j * x_stride)], scratch_ptr1[((j + 1) * x_stride)]);

      accu_int32_0 = AE_ADD32S(accu_int32_0, d_bias);
      MPY_BY_QUANT_MULT_X2_OUT32(accu_int32_0, accu_int32_0, out_multiplier, left_shift, right_shift);
      accu_int32_0 = AE_ADD32S(accu_int32_0, AE_MOVDA32X2(out_zero_bias, out_zero_bias));
      accu_int8x8 = AE_SAT8X4X32_L(accu_int32_0, accu_int32_0);

      *(ae_int8 *)(&out_ptr[((j + 1) * out_stride)]) = AE_MOVINT8_FROMINT8X8(accu_int8x8);
      *(ae_int8 *)(&out_ptr[(j * out_stride)]) = AE_MOVINT8_FROMINT8X8(AE_SEL8X8I(accu_int8x8, accu_int8x8, 19));
    }
    if(j < actual_out_width)
    {
      accu_int32_0 = scratch_ptr1[(j * x_stride)];

      accu_int32_0 = AE_ADD32S(accu_int32_0, d_bias);
      MPY_BY_QUANT_MULT_X2_OUT32(accu_int32_0, accu_int32_0, out_multiplier, left_shift, right_shift);
      accu_int32_0 = AE_ADD32S(accu_int32_0, AE_MOVDA32X2(out_zero_bias, out_zero_bias));
      accu_int8x8 = AE_SAT8X4X32_L(accu_int32_0, accu_int32_0);

      *(ae_int8 *)(&out_ptr[(j * out_stride)]) = AE_MOVINT8_FROMINT8X8(accu_int8x8);
    }
  }
}

#define COPY_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    ae_int8x8 *pae_in = (ae_int8x8 *)(&p_in[itr_kh * kw]); \
    ae_int8x8 *pae_out = (ae_int8x8 *)(&p_out[itr_kh * kw_pad]); \
    ae_int8x8 d_tmp; \
    ae_valign in_a = AE_LA64_PP(pae_in); \
    ae_valign out_a = AE_ZALIGN64(); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 3); itr_kw++) \
    { \
      AE_LA8X8_IP(d_tmp, in_a, pae_in); \
      AE_SA8X8_IP(d_tmp, out_a, pae_out); \
    } \
    if(kw & 7) \
    { \
      AE_LA8X8_IP(d_tmp, in_a, pae_in); \
      ae_int64 d_tmp64 = AE_MOVINT64_FROMINT8X8(d_tmp); \
      d_tmp64 = AE_SRAA64(d_tmp64, 8 * (8 - (kw & 7))); \
      d_tmp64 = AE_SLAA64(d_tmp64, 8 * (8 - (kw & 7))); \
      d_tmp = AE_MOVINT8X8_FROMINT64(d_tmp64); \
      AE_SA8X8_IP(d_tmp, out_a, pae_out); \
    } \
    AE_SA64POS_FP(out_a, pae_out); \
  } \
}

static void xa_nn_conv2d_depthwise_nchw_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32  *p_out_multiplier
  ,const WORD32  *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  UWORD8 input_zero_bias_neg = -input_zero_bias;
  xa_nn_conv2d_depthwise_init
    (p_scratch
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,x_stride
    ,y_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,out_width
    ,8
    ,1
    ,(pVOID)(&input_zero_bias_neg)
    );

  xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
  int itr_ic, itr_cm, itr_oh;
  int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
  int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
  int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
  int rows_to_add, top_pad, bottom_pad, rows_added;
  int input_row;
  const WORD8 *pt_ker;
  const WORD8 *pt_inp;
  WORD8 *p_inp_circ;
  int i;
  WORD8 *p_kernel_padded = (WORD8 *)(p_state->p_scratch);
  p_kernel_padded = (WORD8 *)ALIGN_PTR(p_kernel_padded, 8);
  pWORD32 p_tmp_out = (pWORD32)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
  p_tmp_out = (pWORD32)ALIGN_PTR(p_tmp_out, 16);

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  WORD32 bias = 0;

  /* Initialize whole scratch for padded kernel to padding value, after this
     we only have to copy actual kernel values, padding area should remain
     untouched */
  ae_int8x8 *pae_ker_pad = (ae_int8x8 *)p_kernel_padded;
  for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 3); i++)
  {
    pae_ker_pad[i] = AE_MOVDA8(0);
  }
#pragma loop_count min=1
  for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
  {
    pt_inp = (const WORD8 *)&p_inp[itr_ic*input_height*input_width];

    CIRC_BUF_ADD_ROWS_INIT_WITH_PAD_VAL
      (rows_added
      ,rows_to_add
      ,top_pad
      ,bottom_pad
      ,input_row
      ,input_height
      ,input_width
      ,kernel_height
      ,y_stride
      ,x_padding
      ,y_padding
      ,p_circ_buf
      ,pt_inp
      ,&input_zero_bias_neg
      );

#pragma loop_count min=1
    for(itr_oh = 0; itr_oh < out_height; itr_oh += circ_out_height)
    {
      CIRC_BUF_ADD_ROWS_WITH_PAD_VAL
        (rows_added
        ,rows_to_add
        ,top_pad
        ,bottom_pad
        ,input_row
        ,input_height
        ,input_width
        ,circ_out_height
        ,y_stride
        ,x_padding
        ,y_padding
        ,p_circ_buf
        ,pt_inp
        ,&input_zero_bias_neg
        );

      p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

#pragma loop_count min=1
      for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
      {
        pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
        COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
        bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

        conv2d_nchw_sym8sxasym8s_hf5_convmul
          ((WORD8 *)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
          ,p_kernel_padded
          ,p_inp_circ
          ,bias
          ,p_circ_buf->rows
          ,p_circ_buf->row_offset
          ,kernel_height
          ,kernel_width
          ,XT_MIN(circ_out_height, out_height-itr_oh)
          ,out_width
          ,(input_channels * channels_multiplier)
          ,x_stride
          ,y_stride
          ,input_zero_bias
          ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
          ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
          ,out_zero_bias
          ,p_tmp_out
          );
      }
    }
  }
}

#ifndef AE_MULZB3X3O8X8
/* 2D Convolution implementation */
static inline void conv2d_nhwc_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_ker
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *p_bias
  ,int kernel_height
  ,int kernel_width
  ,int out_height
  ,int out_width
  ,int out_channels
  ,int x_stride
  ,int y_stride
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,pWORD32 __restrict__ p_scratch
  )
{
  WORD32 ker_channels_pad, inp_channels_pad;
  WORD32 i, itr_oh, itr_ch, itr_kw;
  ae_int8x8 *pt_inp0;
  pWORD8 pt_ker;
  pUWORD8 out_ptr0, out_ptr1;
  ae_int8x8 *ae_out_ptr0, *ae_out_ptr1;
  ae_valign out0_a, out1_a;
  ae_int16x4 d_inp00, d_inp01;
  ae_int16x4 d_inp10, d_inp11;
  const ae_int32x4 *pt_bias;
  ae_valignx2 bias_a;
  ae_int32x2 d_acc0, d_acc1, d_acc2, d_acc3;
  ae_int32x2 d_bias0, d_bias1, d_bias2, d_bias3;
  ae_int32x2 d_acc4, d_acc5, d_acc6, d_acc7;
  ae_int8x8 d_acc8x8;

  ker_channels_pad = out_channels;
  inp_channels_pad = (out_channels + 7)&(~7);

#pragma loop_count min=1
  for(itr_oh = 0; itr_oh < (out_height); itr_oh += 2)
  {
    out_ptr0 = (UWORD8 *)(&p_out[itr_oh * out_channels * out_width]);
    out_ptr1 = (UWORD8 *)(&p_out[(itr_oh + 1) * out_channels * out_width]);
    ae_out_ptr0 = (ae_int8x8 *)(out_ptr0);
    ae_out_ptr1 = (ae_int8x8 *)(out_ptr1);
    out0_a = AE_ZALIGN64();
    out1_a = AE_ZALIGN64();
    pt_bias = (const ae_int32x4 *)p_bias;
    bias_a = AE_LA128_PP(pt_bias);
#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < out_channels; itr_ch += 8)
    {
      pt_inp0 = (ae_int8x8 *)p_inp;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh * y_stride * kernel_width * inp_channels_pad);
      pt_ker = (WORD8 *)(&p_ker[itr_ch]);
      d_acc0 = AE_ZERO32();
      d_acc1 = AE_ZERO32();
      d_acc2 = AE_ZERO32();
      d_acc3 = AE_ZERO32();
      d_acc4 = AE_ZERO32();
      d_acc5 = AE_ZERO32();
      d_acc6 = AE_ZERO32();
      d_acc7 = AE_ZERO32();

      ae_int8x8 d_inp0, d_inp1;
      ae_int16x4 d_ker0, d_ker1;
      ae_int8x8 *ptae_ker = (ae_int8x8 *)(&pt_ker[0]);
      ae_valign ker_a = AE_LA64_PP(ptae_ker);
      ae_int8x8 d_ker;
#pragma no_unroll
#pragma loop_count min=1
      for(itr_kw = 0; itr_kw < kernel_height * kernel_width; itr_kw++)
      {
        AE_L8X8_XC(d_inp0, pt_inp0, y_stride * kernel_width * inp_channels_pad);
        AE_L8X8_XC(d_inp1, pt_inp0, inp_channels_pad - y_stride * kernel_width * inp_channels_pad);
        AE_LA8X8_IP(d_ker, ker_a, ptae_ker);
        ptae_ker = (ae_int8x8 *)((WORD8 *)ptae_ker + (ker_channels_pad - 8));
        ker_a = AE_LA64_PP(ptae_ker);
        AE_SUBW8(d_inp00, d_inp01, d_inp0, AE_MOVDA8(-input_zero_bias));
        AE_SUBW8(d_inp10, d_inp11, d_inp1, AE_MOVDA8(-input_zero_bias));
        AE_SUBW8(d_ker0, d_ker1, d_ker, AE_MOVDA8(0));
        AE_MULA16X4(d_acc0, d_acc1, d_inp00, d_ker0);
        AE_MULA16X4(d_acc2, d_acc3, d_inp01, d_ker1);
        AE_MULA16X4(d_acc4, d_acc5, d_inp10, d_ker0);
        AE_MULA16X4(d_acc6, d_acc7, d_inp11, d_ker1);
      }
      AE_LA32X2X2_IP(d_bias0, d_bias1, bias_a, pt_bias);
      AE_LA32X2X2_IP(d_bias2, d_bias3, bias_a, pt_bias);
      d_acc0 = AE_ADD32S(d_acc0, d_bias0);
      d_acc1 = AE_ADD32S(d_acc1, d_bias1);
      d_acc2 = AE_ADD32S(d_acc2, d_bias2);
      d_acc3 = AE_ADD32S(d_acc3, d_bias3);
      d_acc4 = AE_ADD32S(d_acc4, d_bias0);
      d_acc5 = AE_ADD32S(d_acc5, d_bias1);
      d_acc6 = AE_ADD32S(d_acc6, d_bias2);
      d_acc7 = AE_ADD32S(d_acc7, d_bias3);
      
      ae_int32x2 d_hh, d_ll;
      d_hh = AE_SEL32_HH(d_acc0, d_acc4);
      d_ll = AE_SEL32_LL(d_acc0, d_acc4);
      d_acc0 = d_hh;
      d_acc4 = d_ll;
      d_hh = AE_SEL32_HH(d_acc1, d_acc5);
      d_ll = AE_SEL32_LL(d_acc1, d_acc5);
      d_acc1 = d_hh;
      d_acc5 = d_ll;
      d_hh = AE_SEL32_HH(d_acc2, d_acc6);
      d_ll = AE_SEL32_LL(d_acc2, d_acc6);
      d_acc2 = d_hh;
      d_acc6 = d_ll;
      d_hh = AE_SEL32_HH(d_acc3, d_acc7);
      d_ll = AE_SEL32_LL(d_acc3, d_acc7);
      d_acc3 = d_hh;
      d_acc7 = d_ll;

      ae_int16x4 d_acc16_0, d_acc16_1, d_acc16_2, d_acc16_3;
      ae_int16x4 d_acc16_4, d_acc16_5, d_acc16_6, d_acc16_7;
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_0, d_acc0, p_out_multiplier[itr_ch + 0], p_out_shift[itr_ch + 0]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_4, d_acc4, p_out_multiplier[itr_ch + 1], p_out_shift[itr_ch + 1]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_1, d_acc1, p_out_multiplier[itr_ch + 2], p_out_shift[itr_ch + 2]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_5, d_acc5, p_out_multiplier[itr_ch + 3], p_out_shift[itr_ch + 3]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_2, d_acc2, p_out_multiplier[itr_ch + 4], p_out_shift[itr_ch + 4]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_6, d_acc6, p_out_multiplier[itr_ch + 5], p_out_shift[itr_ch + 5]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_3, d_acc3, p_out_multiplier[itr_ch + 6], p_out_shift[itr_ch + 6]);
      MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_7, d_acc7, p_out_multiplier[itr_ch + 7], p_out_shift[itr_ch + 7]);

      ae_int16x4 d_33, d_22, d_11, d_00;
      d_33 = AE_SEL16_7362(d_acc16_0, d_acc16_4);
      d_22 = AE_SEL16_7362(d_acc16_1, d_acc16_5);
      d_11 = AE_SEL16_7362(d_acc16_2, d_acc16_6);
      d_00 = AE_SEL16_7362(d_acc16_3, d_acc16_7);
      d_acc16_0 = AE_SEL16_7632(d_33, d_22);
      d_acc16_1 = AE_SEL16_7632(d_11, d_00);
      d_acc16_2 = AE_SEL16_5410(d_33, d_22);
      d_acc16_3 = AE_SEL16_5410(d_11, d_00);

      d_acc16_0 = AE_ADD16S(d_acc16_0, AE_MOVDA16(out_zero_bias));
      d_acc16_1 = AE_ADD16S(d_acc16_1, AE_MOVDA16(out_zero_bias));

      d_acc8x8 = AE_SAT8X8X16(d_acc16_0, d_acc16_1);
      if(out_channels - itr_ch >= 8)
      {
          AE_SA8X8_IP(d_acc8x8, out0_a, ae_out_ptr0);
          AE_SA64POS_FP(out0_a, ae_out_ptr0);
      }
      else
      {
        /* Reverse outputs in 8x8 to get first output to be stored in 0th element then 1st, 2nd etc. */
        d_acc8x8 = AE_SEL8X8(d_acc8x8, d_acc8x8, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x00010203, 0x04050607)));
#pragma no_unroll
#pragma loop_count min=1
        for(i = 0; i < out_channels - itr_ch; i++)
        {
          *(ae_int8 *)(&out_ptr0[itr_ch + i]) = AE_MOVINT8_FROMINT8X8(d_acc8x8);
          /* Rotate right by 1 element */
          d_acc8x8 = AE_SEL8X8I(d_acc8x8, d_acc8x8, 19);
        }
      }

      d_acc16_2 = AE_ADD16S(d_acc16_2, AE_MOVDA16(out_zero_bias));
      d_acc16_3 = AE_ADD16S(d_acc16_3, AE_MOVDA16(out_zero_bias));

      d_acc8x8 = AE_SAT8X8X16(d_acc16_2, d_acc16_3);
      if(out_height-itr_oh >= 2)
      {
        if(out_channels-itr_ch >= 8)
        {
            AE_SA8X8_IP(d_acc8x8, out1_a, ae_out_ptr1);
            AE_SA64POS_FP(out1_a, ae_out_ptr1);
        }
        else
        {
          /* Reverse outputs in 8x8 to get first output to be stored in 0th element then 1st, 2nd etc. */
          d_acc8x8 = AE_SEL8X8(d_acc8x8, d_acc8x8, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x00010203, 0x04050607)));
#pragma no_unroll
#pragma loop_count min=1
          for(i = 0; i < out_channels - itr_ch; i++)
          {
            *(ae_int8 *)(&out_ptr1[itr_ch + i]) = AE_MOVINT8_FROMINT8X8(d_acc8x8);
            /* Rotate right by 1 element */
            d_acc8x8 = AE_SEL8X8I(d_acc8x8, d_acc8x8, 19);
          }
        }
      }
    }
  }
}
#else /*RI6 core*/

/* 2D Convolution implementation */
static inline void conv2d_nhwc_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_ker
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *p_bias
  ,int kernel_height
  ,int kernel_width
  ,int out_height
  ,int out_width
  ,int out_channels
  ,int x_stride
  ,int y_stride
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,pWORD32 __restrict__ p_scratch
  )
{
  WORD32 ker_channels_pad, inp_channels_pad;
  WORD32 itr_oh, itr_ch, itr_kw;
  ae_int8x16 *pt_inp0, *pt_inp1;
  pWORD8 pt_ker;
  pUWORD8 out_ptr0, out_ptr1;
  ae_int8x16 *ae_out_ptr0, *ae_out_ptr1;
  ae_valignx2 out0_a, out1_a;
  const ae_int32x4 *pt_bias;
  const ae_int32x4 *pt_outm;
  const ae_int32x4 *pt_outs;
  ae_valignx2 bias_a;
  ae_int32x2 outmult01, outmult23, outmult45, outmult67;
  ae_int32x2 d_acc0, d_acc1, d_acc2, d_acc3;
  ae_int32x2 d_acc4, d_acc5, d_acc6, d_acc7;
  ae_int32x2 d_acc8, d_acc9, d_acc10, d_acc11;
  ae_int32x2 d_acc12, d_acc13, d_acc14, d_acc15;
  
  ae_int32x2 d_bias0, d_bias1, d_bias2, d_bias3;
  ae_int32x2 d_bias4, d_bias5, d_bias6, d_bias7;
  ae_int8x8 d_acc8x8_0, d_acc8x8_1;

  inp_channels_pad = (out_channels + 15)&(~15);
  ker_channels_pad = (out_channels + 15)&(~15);

  /*Load bias value in state registers*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-input_zero_bias, 0));
  AE_MOVZBVCDR(biasvc1);

  ae_int8x8 zero8x8 = AE_MOVINT8X8_FROMINT16X4(AE_ZERO16());

/*Special case 3x3 code with HiFi5 RI6 ISA*/
  if(kernel_height == 3 && kernel_width == 3)
  {
    pt_bias = (const ae_int32x4 *)p_bias;
    bias_a = AE_LA128_PP(pt_bias);
#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < out_channels; itr_ch += 8)
    {
      int ysXkwXic = y_stride * kernel_width * inp_channels_pad;
      pt_ker = (WORD8 *)(&p_ker[itr_ch]);
      ae_int8x8 *pt_inp0 = (ae_int8x8 *)p_inp;

      AE_LA32X2X2_IP(d_bias0, d_bias1, bias_a, pt_bias);
      AE_LA32X2X2_IP(d_bias2, d_bias3, bias_a, pt_bias);
      
      AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch);

      ae_int8x8 d_inp000, d_inp010, d_inp020;
      ae_int8x8 d_ker00, d_ker10, d_ker20;
      ae_int8x8 d_ker30, d_ker40, d_ker50; 
      ae_int8x8 d_ker60, d_ker70, d_ker80;
      ae_int8x8 *ptae_ker = (ae_int8x8 *)(&pt_ker[0]);

      AE_L8X8_XP(d_ker00, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker10, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker20, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker30, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker40, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker50, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker60, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker70, ptae_ker, ker_channels_pad);
      AE_L8X8_XP(d_ker80, ptae_ker, ker_channels_pad);

      ae_int32x2 lmult01, lmult23, lmult45, lmult67;
      ae_int32x2 rmult01, rmult23, rmult45, rmult67;
#if TFLITE_SINGLE_ROUNDING
      /*Shifts to match with tensorflow*/
      pt_outs = (const ae_int32x4 *)(&p_out_shift[itr_ch]);
      pt_outm = (const ae_int32x4 *)(&p_out_multiplier[itr_ch]);
      ae_valignx2 outs_a, outm_a;
      outs_a = AE_LA128_PP(pt_outs);
      outm_a = AE_LA128_PP(pt_outm);

      AE_LA32X2X2_IP(lmult01, lmult23, outs_a, pt_outs);
      AE_LA32X2X2_IP(lmult45, lmult67, outs_a, pt_outs);
      AE_LA32X2X2_IP(outmult01, outmult23, outm_a, pt_outm);
      AE_LA32X2X2_IP(outmult45, outmult67, outm_a, pt_outm);

#ifdef AE_TRUNCAV32X2F64S
      ae_int16x4 ls0123 = AE_ADD16(AE_SAT16X4(lmult01, lmult23), AE_MOVDA16(17));
      ae_int16x4 ls4567 = AE_ADD16(AE_SAT16X4(lmult45, lmult67), AE_MOVDA16(17));
      lmult01 = AE_MOVINT32X2_FROMINT16X4(ls0123);
      lmult45 = AE_MOVINT32X2_FROMINT16X4(ls4567);
#endif

      (void)rmult01;
      (void)rmult23;
      (void)rmult45;
      (void)rmult67;
#else /* #if TFLITE_SINGLE_ROUNDING */
      /*Shifts to match with tensorflow*/
      int p_left_shift[8] __attribute__ ((aligned (16)));
      int p_right_shift[8] __attribute__ ((aligned (16)));
      ae_int32x2 ONE = AE_MOVDA32(1);
      ae_int32x2 M_ONE = AE_MOVDA32(0xFFFFFFFF);
      pt_outs = (const ae_int32x4 *)(&p_out_shift[itr_ch]);
      ae_int32x4 *p_lmult = (ae_int32x4 *)(&p_left_shift[0]);
      ae_int32x4 *p_rmult = (ae_int32x4 *)(&p_right_shift[0]);
      ae_valignx2 outs_a;
      outs_a = AE_LA128_PP(pt_outs);

      int i;
      for(i = 0; i < 2; i++)
      {
        ae_int32x2 d_shift0, d_shift1;
        ae_int32x2 d_lsh0, d_lsh1;
        ae_int32x2 d_rsh0, d_rsh1;
        ae_int32x2 temp0, temp1, temp2, temp3;
        AE_LA32X2X2_IP(d_shift0, d_shift1, outs_a, pt_outs);
        /* Invert sign for AE_SRAV32RS */
        AE_MUL2P32X4S(d_shift0, d_shift1, d_shift0, d_shift1, M_ONE, M_ONE);
        d_lsh0 = AE_MIN32(AE_ZERO32(), d_shift0);
        d_lsh1 = AE_MIN32(AE_ZERO32(), d_shift1);
        d_rsh0 = AE_MAX32(AE_ZERO32(), d_shift0);
        d_rsh1 = AE_MAX32(AE_ZERO32(), d_shift1);
        temp0 = AE_SRAV32RS(ONE, d_lsh0);
        temp1 = AE_SRAV32RS(ONE, d_lsh1);
        AE_S32X2X2_IP(temp0, temp1, p_lmult, 16); 
        temp2 = AE_SRAV32RS(M_ONE, -(AE_MOVDA32(31) - d_rsh0));
        temp3 = AE_SRAV32RS(M_ONE, -(AE_MOVDA32(31) - d_rsh1));
        AE_S32X2X2_IP(temp2, temp3, p_rmult, 16); 
      }

      pt_outm = (const ae_int32x4 *)(&p_out_multiplier[itr_ch]);
      ae_valignx2 outm_a = AE_LA128_PP(pt_outm);
      AE_LA32X2X2_IP(outmult01, outmult23, outm_a, pt_outm);
      AE_LA32X2X2_IP(outmult45, outmult67, outm_a, pt_outm);
#endif /* #if TFLITE_SINGLE_ROUNDING */

#pragma loop_count min=1
      for(itr_oh = 0; itr_oh < (out_height); itr_oh++)
      {
        out_ptr0 = (UWORD8 *)(&p_out[itr_ch + itr_oh * out_channels * out_width]);
        ae_out_ptr0 = (ae_int8x16 *)(out_ptr0);
        out0_a = AE_ZALIGN128();
        
        /* Apply first row of kernel */
        AE_L8X8_XC(d_inp000, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp010, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp020, pt_inp0, inp_channels_pad);
        AE_MULZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, d_ker10, d_ker20, d_inp000, d_inp010, d_inp020);

        /* Apply second row of kernel */
        AE_L8X8_XC(d_inp000, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp010, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp020, pt_inp0, inp_channels_pad);
        AE_MULAZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker30, d_ker40, d_ker50, d_inp000, d_inp010, d_inp020);

        /* Apply third row of kernel */
        AE_L8X8_XC(d_inp000, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp010, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp020, pt_inp0, ysXkwXic - 8*inp_channels_pad);
        AE_MULAZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker60, d_ker70, d_ker80, d_inp000, d_inp010, d_inp020);
        
        d_acc0  = AE_ADD32S(d_acc0, d_bias0);
        d_acc1  = AE_ADD32S(d_acc1, d_bias1);
        d_acc2  = AE_ADD32S(d_acc2, d_bias2);
        d_acc3  = AE_ADD32S(d_acc3, d_bias3);

#if !TFLITE_SINGLE_ROUNDING        
        p_lmult = (ae_int32x4 *)(&p_left_shift[0]);
        p_rmult = (ae_int32x4 *)(&p_right_shift[0]);
#endif

        ae_int16x4 out0, out1;
#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_I(lmult01, lmult23, p_lmult, 0);
        AE_L32X2X2_I(rmult01, rmult23, p_rmult, 0);
#endif
        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB_AV(out0, d_acc0, d_acc1, outmult01, outmult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

#if !TFLITE_SINGLE_ROUNDING
        AE_L32X2X2_I(lmult45, lmult67, p_lmult, 16);
        AE_L32X2X2_I(rmult45, rmult67, p_rmult, 16);
#endif

        MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB_AV(out1, d_acc2, d_acc3, outmult45, outmult67, lmult45, lmult67, rmult45, rmult67, out_zero_bias);

        d_acc8x8_0 = AE_SAT8X8X16(out0, out1);
        
        AE_SAV8X8X2_XP(d_acc8x8_0, d_acc8x8_0, out0_a, ae_out_ptr0, XT_MIN(out_channels - itr_ch, 8));
        AE_SA128POS_FP(out0_a, ae_out_ptr0);
      }
    }
  }
  else
  {
#pragma loop_count min=1
    for(itr_oh = 0; itr_oh < (out_height); itr_oh += 2)
    {
      out_ptr0 = (UWORD8 *)(&p_out[itr_oh * out_channels * out_width]);
      out_ptr1 = (UWORD8 *)(&p_out[(itr_oh + 1) * out_channels * out_width]);
      ae_out_ptr0 = (ae_int8x16 *)(out_ptr0);
      ae_out_ptr1 = (ae_int8x16 *)(out_ptr1);
      out0_a = AE_ZALIGN128();
      out1_a = AE_ZALIGN128();
      pt_bias = (const ae_int32x4 *)p_bias;
      bias_a = AE_LA128_PP(pt_bias);
      int ysXkwXic = y_stride * kernel_width * inp_channels_pad;

#pragma loop_count min=1
      for(itr_ch = 0; itr_ch < out_channels; itr_ch += 16)
      {
        pt_inp0 = (ae_int8x16 *)p_inp;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh * ysXkwXic);
        pt_inp1 = (ae_int8x16 *)p_inp;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, itr_ch + (itr_oh + 1) * ysXkwXic);
        pt_ker  = (WORD8 *)(&p_ker[itr_ch]);

        AE_LA32X2X2_IP(d_bias0, d_bias1, bias_a, pt_bias);
        AE_LA32X2X2_IP(d_bias2, d_bias3, bias_a, pt_bias);
        AE_LA32X2X2_IP(d_bias4, d_bias5, bias_a, pt_bias);
        AE_LA32X2X2_IP(d_bias6, d_bias7, bias_a, pt_bias);

        d_acc0  = AE_ZERO32();
        d_acc1  = AE_ZERO32();
        d_acc2  = AE_ZERO32();
        d_acc3  = AE_ZERO32();
        d_acc4  = AE_ZERO32();
        d_acc5  = AE_ZERO32();
        d_acc6  = AE_ZERO32();
        d_acc7  = AE_ZERO32();
        d_acc8  = AE_ZERO32();
        d_acc9  = AE_ZERO32();
        d_acc10 = AE_ZERO32();
        d_acc11 = AE_ZERO32();
        d_acc12 = AE_ZERO32();
        d_acc13 = AE_ZERO32();
        d_acc14 = AE_ZERO32();
        d_acc15 = AE_ZERO32();

        ae_int8x8 d_inp000, d_inp001, d_inp010, d_inp011, d_inp020, d_inp021;
        ae_int8x8 d_inp100, d_inp101, d_inp110, d_inp111, d_inp120, d_inp121;
        ae_int8x8 d_ker00, d_ker01, d_ker10, d_ker11, d_ker20, d_ker21;
        ae_int8x16 *ptae_ker = (ae_int8x16 *)(&pt_ker[0]);

        int loop_count = kernel_height * kernel_width / 3;
#pragma no_unroll 
        for(itr_kw = 0; itr_kw < loop_count; itr_kw++)
        {
          AE_L8X8X2_XC(d_inp000, d_inp001, pt_inp0, inp_channels_pad);
          AE_L8X8X2_XC(d_inp010, d_inp011, pt_inp0, inp_channels_pad);
          AE_L8X8X2_XC(d_inp020, d_inp021, pt_inp0, inp_channels_pad);
          AE_L8X8X2_XC(d_inp100, d_inp101, pt_inp1, inp_channels_pad);
          AE_L8X8X2_XC(d_inp110, d_inp111, pt_inp1, inp_channels_pad);
          AE_L8X8X2_XC(d_inp120, d_inp121, pt_inp1, inp_channels_pad);
          AE_L8X8X2_XP(d_ker00, d_ker01, ptae_ker, ker_channels_pad);
          AE_L8X8X2_XP(d_ker10, d_ker11, ptae_ker, ker_channels_pad);
          AE_L8X8X2_XP(d_ker20, d_ker21, ptae_ker, ker_channels_pad);

          AE_MULAZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, d_ker10, d_ker20, d_inp000, d_inp010, d_inp020);
          AE_MULAZB3X3O8X8(d_acc4, d_acc5, d_acc6, d_acc7, d_ker01, d_ker11, d_ker21, d_inp001, d_inp011, d_inp021);
          AE_MULAZB3X3O8X8(d_acc8, d_acc9, d_acc10, d_acc11, d_ker00, d_ker10, d_ker20, d_inp100, d_inp110, d_inp120);
          AE_MULAZB3X3O8X8(d_acc12, d_acc13, d_acc14, d_acc15, d_ker01, d_ker11, d_ker21, d_inp101, d_inp111, d_inp121);
        }
        for(itr_kw = loop_count * 3; itr_kw < kernel_height * kernel_width; itr_kw++)
        {
          AE_L8X8X2_XC(d_inp000, d_inp001, pt_inp0, inp_channels_pad);
          AE_L8X8X2_XC(d_inp100, d_inp101, pt_inp1, inp_channels_pad);
          AE_L8X8X2_XP(d_ker00, d_ker01, ptae_ker, ker_channels_pad);
          AE_MULAZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, zero8x8, zero8x8, d_inp000, d_inp000, d_inp000);
          AE_MULAZB3X3O8X8(d_acc4, d_acc5, d_acc6, d_acc7, d_ker01, zero8x8, zero8x8, d_inp001, d_inp001, d_inp001);
          AE_MULAZB3X3O8X8(d_acc8, d_acc9, d_acc10, d_acc11, d_ker00, zero8x8, zero8x8, d_inp100, d_inp100, d_inp100);
          AE_MULAZB3X3O8X8(d_acc12, d_acc13, d_acc14, d_acc15, d_ker01, zero8x8, zero8x8, d_inp101, d_inp101, d_inp101);
        }
        
        d_acc0  = AE_ADD32S(d_acc0, d_bias0);
        d_acc1  = AE_ADD32S(d_acc1, d_bias1);
        d_acc2  = AE_ADD32S(d_acc2, d_bias2);
        d_acc3  = AE_ADD32S(d_acc3, d_bias3);
        d_acc4  = AE_ADD32S(d_acc4, d_bias4);
        d_acc5  = AE_ADD32S(d_acc5, d_bias5);
        d_acc6  = AE_ADD32S(d_acc6, d_bias6);
        d_acc7  = AE_ADD32S(d_acc7, d_bias7);
        d_acc8  = AE_ADD32S(d_acc8, d_bias0);
        d_acc9  = AE_ADD32S(d_acc9, d_bias1);
        d_acc10 = AE_ADD32S(d_acc10, d_bias2);
        d_acc11 = AE_ADD32S(d_acc11, d_bias3);
        d_acc12 = AE_ADD32S(d_acc12, d_bias4);
        d_acc13 = AE_ADD32S(d_acc13, d_bias5);
        d_acc14 = AE_ADD32S(d_acc14, d_bias6);
        d_acc15 = AE_ADD32S(d_acc15, d_bias7);
        
        ae_int32x2 d_hh, d_ll;
        d_hh = AE_SEL32_HH(d_acc0, d_acc8);
        d_ll = AE_SEL32_LL(d_acc0, d_acc8);
        d_acc0 = d_hh;
        d_acc8 = d_ll;
        d_hh = AE_SEL32_HH(d_acc1, d_acc9);
        d_ll = AE_SEL32_LL(d_acc1, d_acc9);
        d_acc1 = d_hh;
        d_acc9 = d_ll;
        d_hh = AE_SEL32_HH(d_acc2, d_acc10);
        d_ll = AE_SEL32_LL(d_acc2, d_acc10);
        d_acc2 = d_hh;
        d_acc10 = d_ll;
        d_hh = AE_SEL32_HH(d_acc3, d_acc11);
        d_ll = AE_SEL32_LL(d_acc3, d_acc11);
        d_acc3 = d_hh;
        d_acc11 = d_ll;
        d_hh = AE_SEL32_HH(d_acc4, d_acc12);
        d_ll = AE_SEL32_LL(d_acc4, d_acc12);
        d_acc4 = d_hh;
        d_acc12 = d_ll;
        d_hh = AE_SEL32_HH(d_acc5, d_acc13);
        d_ll = AE_SEL32_LL(d_acc5, d_acc13);
        d_acc5 = d_hh;
        d_acc13 = d_ll;
        d_hh = AE_SEL32_HH(d_acc6, d_acc14);
        d_ll = AE_SEL32_LL(d_acc6, d_acc14);
        d_acc6 = d_hh;
        d_acc14 = d_ll;
        d_hh = AE_SEL32_HH(d_acc7, d_acc15);
        d_ll = AE_SEL32_LL(d_acc7, d_acc15);
        d_acc7 = d_hh;
        d_acc15 = d_ll;

        ae_int16x4 d_acc16_0, d_acc16_1, d_acc16_2, d_acc16_3;
        ae_int16x4 d_acc16_4, d_acc16_5, d_acc16_6, d_acc16_7;
        ae_int16x4 d_acc16_8, d_acc16_9, d_acc16_10, d_acc16_11;
        ae_int16x4 d_acc16_12, d_acc16_13, d_acc16_14, d_acc16_15;
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_0, d_acc0, p_out_multiplier[itr_ch + 0], p_out_shift[itr_ch + 0]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_8, d_acc8, p_out_multiplier[itr_ch + 1], p_out_shift[itr_ch + 1]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_1, d_acc1, p_out_multiplier[itr_ch + 2], p_out_shift[itr_ch + 2]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_9, d_acc9, p_out_multiplier[itr_ch + 3], p_out_shift[itr_ch + 3]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_2, d_acc2, p_out_multiplier[itr_ch + 4], p_out_shift[itr_ch + 4]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_10, d_acc10,p_out_multiplier[itr_ch + 5], p_out_shift[itr_ch + 5]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_3, d_acc3, p_out_multiplier[itr_ch + 6], p_out_shift[itr_ch + 6]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_11, d_acc11,p_out_multiplier[itr_ch + 7], p_out_shift[itr_ch + 7]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_4, d_acc4, p_out_multiplier[itr_ch + 8], p_out_shift[itr_ch + 8]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_12, d_acc12,p_out_multiplier[itr_ch + 9], p_out_shift[itr_ch + 9]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_5, d_acc5, p_out_multiplier[itr_ch + 10], p_out_shift[itr_ch + 10]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_13, d_acc13,p_out_multiplier[itr_ch + 11], p_out_shift[itr_ch + 11]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_6, d_acc6, p_out_multiplier[itr_ch + 12], p_out_shift[itr_ch + 12]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_14, d_acc14,p_out_multiplier[itr_ch + 13], p_out_shift[itr_ch + 13]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_7, d_acc7, p_out_multiplier[itr_ch + 14], p_out_shift[itr_ch + 14]);
        MPY_BY_QUANT_MULT_X2_OUT16_NO_LR_SHIFT(d_acc16_15, d_acc15,p_out_multiplier[itr_ch + 15], p_out_shift[itr_ch + 15]);

        ae_int16x4 d_77, d_66, d_55, d_44, d_33, d_22, d_11, d_00;
        d_77 = AE_SEL16_7362(d_acc16_0, d_acc16_8);
        d_66 = AE_SEL16_7362(d_acc16_1, d_acc16_9);
        d_55 = AE_SEL16_7362(d_acc16_2, d_acc16_10);
        d_44 = AE_SEL16_7362(d_acc16_3, d_acc16_11);
        d_33 = AE_SEL16_7362(d_acc16_4, d_acc16_12);
        d_22 = AE_SEL16_7362(d_acc16_5, d_acc16_13);
        d_11 = AE_SEL16_7362(d_acc16_6, d_acc16_14);
        d_00 = AE_SEL16_7362(d_acc16_7, d_acc16_15);
        d_acc16_0 = AE_SEL16_7632(d_77, d_66);
        d_acc16_1 = AE_SEL16_7632(d_55, d_44);
        d_acc16_2 = AE_SEL16_7632(d_33, d_22);
        d_acc16_3 = AE_SEL16_7632(d_11, d_00);
        d_acc16_4 = AE_SEL16_5410(d_77, d_66);
        d_acc16_5 = AE_SEL16_5410(d_55, d_44);
        d_acc16_6 = AE_SEL16_5410(d_33, d_22);
        d_acc16_7 = AE_SEL16_5410(d_11, d_00);

        d_acc16_0  = AE_ADD16S(d_acc16_0, AE_MOVDA16(out_zero_bias));
        d_acc16_1  = AE_ADD16S(d_acc16_1, AE_MOVDA16(out_zero_bias));
        d_acc16_2  = AE_ADD16S(d_acc16_2, AE_MOVDA16(out_zero_bias));
        d_acc16_3  = AE_ADD16S(d_acc16_3, AE_MOVDA16(out_zero_bias));

        d_acc8x8_0 = AE_SAT8X8X16(d_acc16_0, d_acc16_1);
        d_acc8x8_1 = AE_SAT8X8X16(d_acc16_2, d_acc16_3);
        if(out_channels - itr_ch >= 16)
        {
            AE_SA8X8X2_IP(d_acc8x8_0, d_acc8x8_1, out0_a, ae_out_ptr0);
        }
        else
        {        
          AE_SAV8X8X2_XP(d_acc8x8_0, d_acc8x8_1, out0_a, ae_out_ptr0, out_channels - itr_ch);
        }
        AE_SA128POS_FP(out0_a, ae_out_ptr0);

        d_acc16_4  = AE_ADD16S(d_acc16_4, AE_MOVDA16(out_zero_bias));
        d_acc16_5  = AE_ADD16S(d_acc16_5, AE_MOVDA16(out_zero_bias));
        d_acc16_6  = AE_ADD16S(d_acc16_6, AE_MOVDA16(out_zero_bias));
        d_acc16_7  = AE_ADD16S(d_acc16_7, AE_MOVDA16(out_zero_bias));

        d_acc8x8_0 = AE_SAT8X8X16(d_acc16_4, d_acc16_5);
        d_acc8x8_1 = AE_SAT8X8X16(d_acc16_6, d_acc16_7);
        if(out_height-itr_oh >= 2)
        {
          if(out_channels-itr_ch >= 16)
          {
            AE_SA8X8X2_IP(d_acc8x8_0, d_acc8x8_1, out1_a, ae_out_ptr1);
          }
          else
          {
            AE_SAV8X8X2_XP(d_acc8x8_0, d_acc8x8_1, out1_a, ae_out_ptr1, out_channels - itr_ch);
          }
          AE_SA128POS_FP(out1_a, ae_out_ptr1);
        }
        AE_SA128POS_FP(out1_a, ae_out_ptr1);
      }
    }
  }
}
#endif

static void xa_nn_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32  *p_out_multiplier
  ,const WORD32  *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  UWORD8 input_zero_bias_neg = -input_zero_bias;
  xa_nn_conv2d_depthwise_init
    (p_scratch
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,x_stride
    ,y_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,out_width
    ,8
    ,0
    ,(pVOID)(&input_zero_bias_neg)
    );

  xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = p_state;
  int itr_ow;
  int cols_to_add, left_pad, right_pad, cols_added;
  int input_col;
  const WORD8 *pt_inp;
  pWORD8 p_inp_circ;
  const WORD8 *pt_ker = p_kernel;

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  pt_inp = (const WORD8 *)p_inp;

#ifdef AE_MULZB3X3O8X8
  /* copy kernel to scratch to resolve non-alignment scenarios.
   * make the depth of the kernel(out_channels) to a multiple of 16 if required and use it from there */
  int out_channels = input_channels * channels_multiplier;
  const ae_int8x16 *ptae_ker_in;
  ae_int8x16 *ptae_ker_out;
  ae_valignx2 in_a;
  ae_int8x8 d_ker0, d_ker1;
  int itr, itr_ic;
  ptae_ker_in = (const ae_int8x16 *)p_kernel;
  ptae_ker_out = (ae_int8x16 *)p_circ_buf->p_end;
  in_a = AE_LA128_PP(ptae_ker_in);
  for(itr = 0; itr < kernel_height * kernel_width; itr++)
  {
    for(itr_ic = 0; itr_ic < out_channels; itr_ic+=16)
    {
      AE_LAV8X8X2_XP(d_ker0, d_ker1, in_a, ptae_ker_in, XT_MIN(out_channels - itr_ic, 16));
      AE_S8X8X2_IP(d_ker0, d_ker1, ptae_ker_out, 16);
    }
  }
  pt_ker = (const WORD8 *)p_circ_buf->p_end;
#endif

  CIRC_BUF_ADD_COLS_INIT_WITH_PAD_VAL
    (cols_added
    ,cols_to_add
    ,left_pad
    ,right_pad
    ,input_col
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_width
    ,channels_multiplier
    ,x_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,p_circ_buf
    ,pt_inp
    ,&input_zero_bias_neg
    );

#pragma loop_count min=1
  for(itr_ow = 0; itr_ow < out_width; itr_ow++)
  {
    CIRC_BUF_ADD_COLS_WITH_PAD_VAL
      (cols_added
      ,cols_to_add
      ,left_pad
      ,right_pad
      ,input_col
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,p_circ_buf
      ,pt_inp
      ,&input_zero_bias_neg
      );

    p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

    conv2d_nhwc_per_chan_sym8sxasym8s
      ((pWORD8)(&p_out[itr_ow*input_channels*channels_multiplier])
      ,pt_ker
      ,p_inp_circ
      ,p_bias
      ,kernel_height
      ,kernel_width
      ,out_height
      ,out_width
      ,(input_channels * channels_multiplier)
      ,x_stride
      ,y_stride
      ,input_zero_bias
      ,p_out_multiplier
      ,p_out_shift
      ,out_zero_bias
      ,p_scratch
      );
  }
}

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE

#define KH_3X3 3
#define KW_3X3 3

static xa_nn_conv2d_dw_k3x3_state_t* xa_nn_conv2d_depthwise_init_nhwc_k3x3
  (pVOID p_scratch
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD32 *__restrict__ p_bias
  ,WORD32 input_height
  ,WORD32 input_channels
  ,WORD32 out_height
  ,WORD32 y_stride
  ,WORD32 y_padding 
  ,WORD32 input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  )
{
  int i;
  xa_nn_conv2d_dw_k3x3_state_t* p_state;
  pWORD8 p_mem = (pWORD8)ALIGN_PTR(p_scratch, ALIGNMENT_16);

  /* State structure */
  p_state = (xa_nn_conv2d_dw_k3x3_state_t *)p_mem;
  p_mem += ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_k3x3_state_t), ALIGNMENT_16);
  memset(p_state, 0, sizeof(xa_nn_conv2d_dw_k3x3_state_t));

  /* Initial accumulator values: output bias */
  p_state->p_accu = (WORD32 *)p_mem;
  p_mem += ALIGNED_SIZE(input_channels * sizeof(WORD32), ALIGNMENT_16);
  p_state->p_accu_zero_point = (WORD32 *)p_mem;
  p_mem += ALIGNED_SIZE(input_channels * sizeof(WORD32), ALIGNMENT_16);
  /* Output multipliers: l_mult, out_multiplier, r_mult */
  p_state->p_scale_multipliers = (WORD32 *)p_mem;
  p_mem += ALIGNED_SIZE(3 * input_channels * sizeof(WORD32), ALIGNMENT_16);
  /* Dummy input buffer: 3 rows */ 
  p_state->p_dummy_inp = p_mem;
  p_mem += ALIGNED_SIZE(KH_3X3 * input_channels * sizeof(WORD8), ALIGNMENT_16);
  /* Rearragned kernel */
  p_state->p_kernel_rearranged = p_mem;
  p_mem += ALIGNED_SIZE(KH_3X3 * KW_3X3 * input_channels * sizeof(WORD8), ALIGNMENT_16);
  /* Rearragned NCHW kernel */
  p_state->p_kernel_nchw = p_mem;
  p_mem += ALIGNED_SIZE((KH_3X3 + 1) * (KW_3X3 + 1) * input_channels * sizeof(WORD8), ALIGNMENT_16);

  /* Initialize the accumulator and output quantization multipliers */
  for(i = 0; i < input_channels; i += 4)
  {
    (p_state->p_accu)[i + 0] = p_bias[i + 0];
    (p_state->p_accu)[i + 1] = p_bias[i + 1];
    (p_state->p_accu)[i + 2] = p_bias[i + 2];
    (p_state->p_accu)[i + 3] = p_bias[i + 3];


#if TFLITE_SINGLE_ROUNDING
    (p_state->p_scale_multipliers)[i * 3 + 0] = p_out_shift[i + 0];

    (p_state->p_scale_multipliers)[i * 3 + 1] = p_out_shift[i + 1];

    (p_state->p_scale_multipliers)[i * 3 + 2] = p_out_shift[i + 2];

    (p_state->p_scale_multipliers)[i * 3 + 3] = p_out_shift[i + 3];
#else /* #if TFLITE_SINGLE_ROUNDING */
    WORD32 left_shift, right_shift;

    left_shift = p_out_shift[i + 0] < 0 ? 0 : p_out_shift[i + 0]; 
    (p_state->p_scale_multipliers)[i * 3 + 0] = (1 << left_shift);

    left_shift = p_out_shift[i + 1] < 0 ? 0 : p_out_shift[i + 1]; 
    (p_state->p_scale_multipliers)[i * 3 + 1] = (1 << left_shift);

    left_shift = p_out_shift[i + 2] < 0 ? 0 : p_out_shift[i + 2]; 
    (p_state->p_scale_multipliers)[i * 3 + 2] = (1 << left_shift);

    left_shift = p_out_shift[i + 3] < 0 ? 0 : p_out_shift[i + 3]; 
    (p_state->p_scale_multipliers)[i * 3 + 3] = (1 << left_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */

    (p_state->p_scale_multipliers)[i * 3 + 4] = p_out_multiplier[i + 0];
    (p_state->p_scale_multipliers)[i * 3 + 5] = p_out_multiplier[i + 1];
    (p_state->p_scale_multipliers)[i * 3 + 6] = p_out_multiplier[i + 2];
    (p_state->p_scale_multipliers)[i * 3 + 7] = p_out_multiplier[i + 3];

#if TFLITE_SINGLE_ROUNDING
    (p_state->p_scale_multipliers)[i * 3 + 8] = p_out_shift[i + 0];

    (p_state->p_scale_multipliers)[i * 3 + 9] = p_out_shift[i + 1];

    (p_state->p_scale_multipliers)[i * 3 + 10] = p_out_shift[i + 2];

    (p_state->p_scale_multipliers)[i * 3 + 11] = p_out_shift[i + 3];
#else /* #if TFLITE_SINGLE_ROUNDING */
    right_shift = p_out_shift[i + 0] > 0 ? 0 : -p_out_shift[i + 0]; 
    (p_state->p_scale_multipliers)[i * 3 + 8] = (0xFFFFFFFF << (31 - right_shift));

    right_shift = p_out_shift[i + 1] > 0 ? 0 : -p_out_shift[i + 1]; 
    (p_state->p_scale_multipliers)[i * 3 + 9] = (0xFFFFFFFF << (31 - right_shift));

    right_shift = p_out_shift[i + 2] > 0 ? 0 : -p_out_shift[i + 2]; 
    (p_state->p_scale_multipliers)[i * 3 + 10] = (0xFFFFFFFF << (31 - right_shift));

    right_shift = p_out_shift[i + 3] > 0 ? 0 : -p_out_shift[i + 3]; 
    (p_state->p_scale_multipliers)[i * 3 + 11] = (0xFFFFFFFF << (31 - right_shift));
#endif /* #if TFLITE_SINGLE_ROUNDING */
  }

  for(i = 0; i < input_channels; i++)
  {
    WORD32 accu_with_zero_point;
    accu_with_zero_point = p_bias[i];
    accu_with_zero_point -= input_zero_bias * p_kernel[i];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + 1 * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + 2 * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (KW_3X3 + 0) * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (KW_3X3 + 1) * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (KW_3X3 + 2) * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (2 * KW_3X3 + 0) * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (2 * KW_3X3 + 1) * input_channels];
    accu_with_zero_point -= input_zero_bias * p_kernel[i + (2 * KW_3X3 + 2) * input_channels];

    (p_state->p_accu_zero_point)[i] = accu_with_zero_point; 
  }

  const WORD8* p_ker0 = p_kernel;
  const WORD8* p_ker1 = (p_kernel + input_channels);
  const WORD8* p_ker2 = (p_ker1   + input_channels);
  ae_int32* p_out = (ae_int32*)(p_state->p_kernel_rearranged);
  ae_int8x16* p_out_nchw = (ae_int8x16*)(p_state->p_kernel_nchw);

  for(i = 0; i < input_channels; i += 4)
  {
    ae_int16x4 ker00, ker01, ker02; 
    ae_int16x4 ker10, ker11, ker12; 
    ae_int16x4 ker20, ker21, ker22; 

    ae_int8x8 data00, data01;
    ae_int8x8 data10, data11;
    ae_int8x8 data20, data21;

    ker20 = AE_L8X4S_X(p_ker0, 2 * KW_3X3 * input_channels);
    ker10 = AE_L8X4S_X(p_ker0, 1 * KW_3X3 * input_channels);
    AE_L8X4S_IP(ker00, p_ker0, 4);

    ker21 = AE_L8X4S_X(p_ker1, 2 * KW_3X3 * input_channels);
    ker11 = AE_L8X4S_X(p_ker1, 1 * KW_3X3 * input_channels);
    AE_L8X4S_IP(ker01, p_ker1, 4);

    ker22 = AE_L8X4S_X(p_ker2, 2 * KW_3X3 * input_channels);
    ker12 = AE_L8X4S_X(p_ker2, 1 * KW_3X3 * input_channels);
    AE_L8X4S_IP(ker02, p_ker2, 4);

    ae_int8x8 out32_0;
    PACK_32X2(out32_0, ker00, ker01);
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);
    AE_S32_L_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);

    PACK_32X2(out32_0, ker02, ker10);
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);
    AE_S32_L_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);

    PACK_32X2(out32_0, ker11, ker12);
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);
    AE_S32_L_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);

    PACK_32X2(out32_0, ker20, ker21);
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);
    AE_S32_L_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);

    PACK_32X2(out32_0, ker22, AE_ZERO16());
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), p_out, 4);

    INTERLEAVE_4(data00, data01, ker00, ker01, ker02, AE_ZERO16());
    INTERLEAVE_4(data10, data11, ker10, ker11, ker12, AE_ZERO16());
    INTERLEAVE_4(data20, data21, ker20, ker21, ker22, AE_ZERO16());

    AE_S8X8X2_IP(data00, data01, p_out_nchw, 16);
    AE_S8X8X2_IP(data10, data11, p_out_nchw, 16);
    AE_S8X8X2_IP(data20, data21, p_out_nchw, 16);

  }

  /* Set the dummy input to zero_bias values */
  memset(p_state->p_dummy_inp, (WORD8)input_zero_bias, KH_3X3 * input_channels * sizeof(WORD8)); 

  /* Calculate loop counters for output height */
  int itr_oh;

  for(itr_oh = 0; itr_oh < out_height; )
  {
    int y_start = itr_oh * y_stride;
    int y_stop = y_start + KH_3X3 - 1;
    int y_stop_h4 = (itr_oh + 3) * y_stride  + KW_3X3 - 1;

    if(y_stop < y_padding)
    {
      p_state->top_padded_region_output++;
      itr_oh++;
    }
    else if(y_start >= (y_padding + input_height))
    {
      p_state->bottom_padded_region_output++;
      itr_oh++;
    }
    else
    {
      /* One or more valid input rows */
      if((y_start >= y_padding) &&
         (y_stop_h4 < (y_padding + input_height)) &&
         ((itr_oh + 3) < out_height) &&
         1)
      {
        p_state->six_input_row_output++;
        itr_oh += 4;
      }
      else
      {
        if(y_stop == y_padding)
        {
          p_state->top_single_input_row_output++;
        }
        else if(y_start == (y_padding + input_height - 1))
        {
          p_state->bottom_single_input_row_output++;
        }
        else if((y_start == (y_padding - 1)) && (input_height == 1))
        {
          p_state->middle_single_input_row_output++;
        }
        else if(y_start == (y_padding + input_height - 2))
        {
          p_state->bottom_two_input_row_output++;
        }
        else if(y_start == (y_padding - 1))
        {
          p_state->top_two_input_row_output++;
        }
        else
        {
          p_state->three_input_row_output++;
        }
        itr_oh++;
      }
    }
  }

  return p_state;
}

static void process_padded_region_output_row
  (pWORD8 __restrict__ p_out
  ,const WORD32 * p_bias
  ,WORD32  input_channels
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_ch;
  ae_valignx2 bias_a;
  ae_int32x2 d_acc01, d_acc23;
  const WORD32 *p_scale_multipliers = p_state->p_scale_multipliers;

  bias_a = AE_LA128_PP(p_bias);

  for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
  {
    AE_LA32X2X2_IP(d_acc01, d_acc23, bias_a, (ae_int32x4*)p_bias);

    /* Quantize */
    ae_int32x2 lmult01, lmult23;
    ae_int32x2 mult01, mult23;
    ae_int32x2 rmult01, rmult23;
    AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

    ae_int16x4 out_0;
    MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc01, d_acc23, \
        mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

    /* Pack and store output */
    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, AE_ZERO16());
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out, 4);
  }
}

static void process_padded_region_output_vplane
  (pWORD8 __restrict__ p_out
  ,const WORD32 * p_bias
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_oh;
  for(itr_oh = 0; itr_oh < out_height; itr_oh++)
  {
    process_padded_region_output_row(p_out + itr_oh * out_channels * out_width
        ,p_bias
        ,out_channels
        ,out_zero_bias
        ,p_state
        );
  }
}

static void process_single_input_row
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,WORD32  input_channels
  ,WORD32  input_zero_bias
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_ch;
  const WORD8 * p_inp0, *p_inp1, *p_inp2;
  const WORD8 * p_ker0, *p_ker1, *p_ker2;

  const WORD32 *p_scale_multipliers = p_state->p_scale_multipliers;

  const WORD32 *__restrict__ p_accu = p_state->p_accu;

  /* Set up input and kernel pointers */
  p_inp0 = p_state->p_inp0;
  p_inp1 = p_state->p_inp1;
  p_inp2 = p_state->p_inp2;

  p_ker0 = p_kernel;
  p_ker1 = (p_kernel + input_channels);
  p_ker2 = (p_ker1   + input_channels);

  for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
  {
    ae_int32x2 d_acc23, d_acc10;
    ae_int16x4 inp00, inp01, inp02; 
    ae_int16x4 ker00, ker01, ker02; 
    /* Initialize accumulators with bias */
    AE_L32X2X2_IP(d_acc23, d_acc10, (ae_int32x4 *)p_accu, 16);

    /* Load input */
    AE_L8X4S_IP(inp00, p_inp0, 4);
    AE_L8X4S_IP(inp01, p_inp1, 4);
    AE_L8X4S_IP(inp02, p_inp2, 4);

    inp00 = AE_SUB16S(inp00, AE_MOVDA16(input_zero_bias));
    inp01 = AE_SUB16S(inp01, AE_MOVDA16(input_zero_bias));
    inp02 = AE_SUB16S(inp02, AE_MOVDA16(input_zero_bias));

    /* Load kernel */
    AE_L8X4S_IP(ker00, p_ker0, 4);
    AE_L8X4S_IP(ker01, p_ker1, 4);
    AE_L8X4S_IP(ker02, p_ker2, 4);

    /* Multiply and accumulate */
    AE_MULA16X4(d_acc23, d_acc10, inp00, ker00);
    AE_MULA16X4(d_acc23, d_acc10, inp01, ker01);
    AE_MULA16X4(d_acc23, d_acc10, inp02, ker02);

    /* Quantize */
    ae_int32x2 lmult01, lmult23;
    ae_int32x2 mult01, mult23;
    ae_int32x2 rmult01, rmult23;
    AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

    ae_int16x4 out_0;
    MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc23, d_acc10, \
        mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

    /* Pack and store output */
    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, AE_ZERO16());
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out, 4);
  }
}

static void process_two_input_rows
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,WORD32  input_channels
  ,WORD32  input_zero_bias
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_ch;
  WORD32  kernel_width = 3;
  const WORD8 *p_inp0, *p_inp1, *p_inp2;
  const WORD8 *p_ker0, *p_ker1, *p_ker2;
  int inp0_offset, inp1_offset, inp2_offset;

  const WORD32 *p_scale_multipliers = p_state->p_scale_multipliers;

  const WORD32 *__restrict__ p_accu = p_state->p_accu;

  /* Set up input and kernel pointers */
  p_inp0 = p_state->p_inp0;
  p_inp1 = p_state->p_inp1;
  p_inp2 = p_state->p_inp2;

  inp0_offset = p_state->inp0_offset;
  inp1_offset = p_state->inp1_offset;
  inp2_offset = p_state->inp2_offset;

  p_ker0 = p_kernel;
  p_ker1 = (p_kernel + input_channels);
  p_ker2 = (p_ker1   + input_channels);

  for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
  {
    ae_int32x2 d_acc23, d_acc10;

    ae_int16x4 inp00, inp01, inp02; 
    ae_int16x4 inp10, inp11, inp12; 

    ae_int16x4 ker00, ker01, ker02; 
    ae_int16x4 ker10, ker11, ker12; 

    /* Initialize accumulators with bias */
    AE_L32X2X2_IP(d_acc23, d_acc10, (ae_int32x4 *)p_accu, 16);

    /* Load input */
    inp10 = AE_L8X4S_X(p_inp0, inp0_offset);
    AE_L8X4S_IP(inp00, p_inp0, 4);
    inp11 = AE_L8X4S_X(p_inp1, inp1_offset);
    AE_L8X4S_IP(inp01, p_inp1, 4);
    inp12 = AE_L8X4S_X(p_inp2, inp2_offset);
    AE_L8X4S_IP(inp02, p_inp2, 4);

    inp00 = AE_SUB16S(inp00, AE_MOVDA16(input_zero_bias));
    inp01 = AE_SUB16S(inp01, AE_MOVDA16(input_zero_bias));
    inp02 = AE_SUB16S(inp02, AE_MOVDA16(input_zero_bias));
    inp10 = AE_SUB16S(inp10, AE_MOVDA16(input_zero_bias));
    inp11 = AE_SUB16S(inp11, AE_MOVDA16(input_zero_bias));
    inp12 = AE_SUB16S(inp12, AE_MOVDA16(input_zero_bias));

    /* Load kernel */
    ker10 = AE_L8X4S_X(p_ker0, kernel_width * input_channels);
    AE_L8X4S_IP(ker00, p_ker0, 4);
    ker11 = AE_L8X4S_X(p_ker1, kernel_width * input_channels);
    AE_L8X4S_IP(ker01, p_ker1, 4);
    ker12 = AE_L8X4S_X(p_ker2, kernel_width * input_channels);
    AE_L8X4S_IP(ker02, p_ker2, 4);

    /* Multiply and accumulate */
    AE_MULA16X4(d_acc23, d_acc10, inp00, ker00);
    AE_MULA16X4(d_acc23, d_acc10, inp01, ker01);
    AE_MULA16X4(d_acc23, d_acc10, inp02, ker02);
    AE_MULA16X4(d_acc23, d_acc10, inp10, ker10);
    AE_MULA16X4(d_acc23, d_acc10, inp11, ker11);
    AE_MULA16X4(d_acc23, d_acc10, inp12, ker12);

    /* Quantize */
    ae_int32x2 lmult01, lmult23;
    ae_int32x2 mult01, mult23;
    ae_int32x2 rmult01, rmult23;
    AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

    ae_int16x4 out_0;
    MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc23, d_acc10, \
        mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

    /* Pack and store output */
    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, AE_ZERO16());
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out, 4);
  }
}

static inline void increment_input_pointer
    (xa_nn_conv2d_dw_k3x3_state_t *p_state
    ,int rows
    )
{
  p_state->p_inp0 += rows * p_state->inp0_offset;
  p_state->p_inp1 += rows * p_state->inp1_offset;
  p_state->p_inp2 += rows * p_state->inp2_offset;
}

static pWORD8 process_six_input_rows
  (pWORD8 __restrict__ p_out
  ,WORD32  input_channels
  ,WORD32  out_width 
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t * __restrict__ p_state
  )
{
  int itr_oh, itr_ch;
  const WORD8 * __restrict__ p_inp0, *__restrict__ p_inp1, *__restrict__ p_inp2;
  const ae_int8x16 * __restrict__ p_ker0;
  int inp0_offset, inp1_offset, inp2_offset;

  WORD8 * __restrict__ p_out0;

  const WORD32 * __restrict__ p_scale_multipliers;
  const WORD8 * __restrict__ p_kernel;
  const WORD32 *__restrict__ p_accu; 

  inp0_offset = p_state->inp0_offset;
  inp1_offset = p_state->inp1_offset;
  inp2_offset = p_state->inp2_offset;


  for(itr_oh = 0; itr_oh < p_state->six_input_row_output; itr_oh++)
  {
    p_scale_multipliers = p_state->p_scale_multipliers;
    p_kernel = p_state->p_kernel_nchw;
    p_accu = p_state->p_accu_zero_point;

    p_out0 = p_out;

    /* Set up input and kernel pointers */
    p_inp0 = p_state->p_inp0;
    p_inp1 = p_state->p_inp1;
    p_inp2 = p_state->p_inp2;

    p_ker0 = (ae_int8x16 *)p_kernel;

#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
    {
      ae_int32x2 d_acc01_0, d_acc23_0;
      ae_int32x2 d_acc01_1, d_acc23_1;
      ae_int32x2 d_acc01_2, d_acc23_2;
      ae_int32x2 d_acc01_3, d_acc23_3;

      ae_int32x2 acc_ch0_01, acc_ch0_23;
      ae_int32x2 acc_ch1_01, acc_ch1_23;
      ae_int32x2 acc_ch2_01, acc_ch2_23;
      ae_int32x2 acc_ch3_01, acc_ch3_23;

      ae_int16x4 inp00, inp10, inp20, inp30, inp40, inp50; 
      ae_int16x4 inp01, inp11, inp21, inp31, inp41, inp51; 
      ae_int16x4 inp02, inp12, inp22, inp32, inp42, inp52; 

      ae_int8x8 data00, data01;
      ae_int8x8 data10, data11;
      ae_int8x8 data20, data21;
      ae_int8x8 data30, data31;
      ae_int8x8 data40, data41;
      ae_int8x8 data50, data51;

      ae_int8x8 ker00, ker01; 
      ae_int8x8 ker10, ker11; 
      ae_int8x8 ker20, ker21; 

      /* Initialize accumulators with bias */
      ae_int32x2 temp0, temp1;
      AE_L32X2X2_IP(temp0, temp1, (ae_int32x4 *)p_accu, 16);
#if 0
      acc_ch0_01 = acc_ch0_23 = AE_SEL32_HH(temp0, temp0);
      acc_ch1_01 = acc_ch1_23 = AE_SEL32_LL(temp0, temp0);
      acc_ch2_01 = acc_ch2_23 = AE_SEL32_HH(temp1, temp1);
      acc_ch3_01 = acc_ch3_23 = AE_SEL32_LL(temp1, temp1);
#else
      ae_int8x8 ch0_01, ch0_23;
      ae_int8x8 ch1_01, ch1_23;
      ae_int8x8 ch2_01, ch2_23;
      ae_int8x8 ch3_01, ch3_23;

      AE_DSEL8X8(ch0_01, ch1_01, AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch0_23, ch1_23, AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch2_01, ch3_01, AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch2_23, ch3_23, AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));

      acc_ch0_01 = AE_MOVINT32X2_FROMINT8X8(ch0_01);
      acc_ch0_23 = AE_MOVINT32X2_FROMINT8X8(ch0_23);
      acc_ch1_01 = AE_MOVINT32X2_FROMINT8X8(ch1_01);
      acc_ch1_23 = AE_MOVINT32X2_FROMINT8X8(ch1_23);
      acc_ch2_01 = AE_MOVINT32X2_FROMINT8X8(ch2_01);
      acc_ch2_23 = AE_MOVINT32X2_FROMINT8X8(ch2_23);
      acc_ch3_01 = AE_MOVINT32X2_FROMINT8X8(ch3_01);
      acc_ch3_23 = AE_MOVINT32X2_FROMINT8X8(ch3_23);
#endif

      /* Load input */
      AE_L8X4S_XP(inp00, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp10, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp20, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp30, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp40, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp50, p_inp0, - 5 *inp0_offset + 4);

      AE_L8X4S_XP(inp01, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp11, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp21, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp31, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp41, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp51, p_inp1, -5 * inp1_offset + 4);

      AE_L8X4S_XP(inp02, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp12, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp22, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp32, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp42, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp52, p_inp2, -5 * inp2_offset + 4);

      INTERLEAVE_3(data00, data01, inp00, inp01, inp02);
      INTERLEAVE_3(data10, data11, inp10, inp11, inp12);
      INTERLEAVE_3(data20, data21, inp20, inp21, inp22);
      INTERLEAVE_3(data30, data31, inp30, inp31, inp32);
      INTERLEAVE_3(data40, data41, inp40, inp41, inp42);
      INTERLEAVE_3(data50, data51, inp50, inp51, inp52);

      /* Load kernel */
      AE_L8X8X2_IP(ker00, ker01, p_ker0, 16); // ch0, ch1, ch2, ch3
      AE_L8X8X2_IP(ker10, ker11, p_ker0, 16);
      AE_L8X8X2_IP(ker20, ker21, p_ker0, 16);

      /* Multiply and accumulate */
      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data00, data10, data20, data30, ker00); // ch0, ch1 row0
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data01, data11, data21, data31, ker01); // ch2, ch3 row0

      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data10, data20, data30, data40, ker10);
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data11, data21, data31, data41, ker11);

      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data20, data30, data40, data50, ker20);
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data21, data31, data41, data51, ker21);

      /* Quantize */
      ae_int32x2 lmult01, lmult23;
      ae_int32x2 mult01, mult23;
      ae_int32x2 rmult01, rmult23;
      AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
      AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
      AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

#if 0
      d_acc01_0 = AE_SEL32_HH(acc_ch0_01, acc_ch1_01);
      d_acc23_0 = AE_SEL32_HH(acc_ch2_01, acc_ch3_01);
      d_acc01_1 = AE_SEL32_LL(acc_ch0_01, acc_ch1_01);
      d_acc23_1 = AE_SEL32_LL(acc_ch2_01, acc_ch3_01);
      d_acc01_2 = AE_SEL32_HH(acc_ch0_23, acc_ch1_23);
      d_acc23_2 = AE_SEL32_HH(acc_ch2_23, acc_ch3_23);
      d_acc01_3 = AE_SEL32_LL(acc_ch0_23, acc_ch1_23);
      d_acc23_3 = AE_SEL32_LL(acc_ch2_23, acc_ch3_23);
#else
      DSEL32X4_HHLL(d_acc01_0, d_acc01_1, acc_ch0_01, acc_ch1_01); 
      DSEL32X4_HHLL(d_acc23_0, d_acc23_1, acc_ch2_01, acc_ch3_01); 
      DSEL32X4_HHLL(d_acc01_2, d_acc01_3, acc_ch0_23, acc_ch1_23); 
      DSEL32X4_HHLL(d_acc23_2, d_acc23_3, acc_ch2_23, acc_ch3_23); 
#endif

      ae_int16x4 out_0, out_1, out_2, out_3;
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc01_0, d_acc23_0, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_1, d_acc01_1, d_acc23_1, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_2, d_acc01_2, d_acc23_2, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_3, d_acc01_3, d_acc23_3, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

      AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

      /* Pack and store output */
      ae_int8x8 out32_0, out32_1;
      PACK_32X2(out32_0, out_0, out_1);
      PACK_32X2(out32_1, out_2, out_3);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_out0, -3 * input_channels * out_width + 4);
    }

    increment_input_pointer(p_state, 4);
    p_out += 4 * input_channels * out_width;
  }
  return p_out;
}

//supports only y_stride = 2 for now
static pWORD8 process_six_input_rows_ystride
  (pWORD8 __restrict__ p_out
  ,WORD32  input_channels
  ,WORD32  y_stride 
  ,WORD32  out_width 
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t * __restrict__ p_state
  )
{
  int itr_oh, itr_ch;
  const WORD8 * __restrict__ p_inp0, *__restrict__ p_inp1, *__restrict__ p_inp2;
  const ae_int8x16 * __restrict__ p_ker0;
  int inp0_offset, inp1_offset, inp2_offset;

  WORD8 * __restrict__ p_out0;

  const WORD32 * __restrict__ p_scale_multipliers;
  const WORD8 * __restrict__ p_kernel;
  const WORD32 *__restrict__ p_accu; 

  inp0_offset = p_state->inp0_offset;
  inp1_offset = p_state->inp1_offset;
  inp2_offset = p_state->inp2_offset;

  for(itr_oh = 0; itr_oh < p_state->six_input_row_output; itr_oh++)
  {
    p_scale_multipliers = p_state->p_scale_multipliers;
    p_kernel = p_state->p_kernel_nchw;
    p_accu = p_state->p_accu_zero_point;

    p_out0 = p_out;

    /* Set up input and kernel pointers */
    p_inp0 = p_state->p_inp0;
    p_inp1 = p_state->p_inp1;
    p_inp2 = p_state->p_inp2;

    p_ker0 = (ae_int8x16 *)p_kernel;

#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
    {
      ae_int32x2 d_acc01_0, d_acc23_0;
      ae_int32x2 d_acc01_1, d_acc23_1;
      ae_int32x2 d_acc01_2, d_acc23_2;
      ae_int32x2 d_acc01_3, d_acc23_3;

      ae_int32x2 acc_ch0_01, acc_ch0_23;
      ae_int32x2 acc_ch1_01, acc_ch1_23;
      ae_int32x2 acc_ch2_01, acc_ch2_23;
      ae_int32x2 acc_ch3_01, acc_ch3_23;

      ae_int16x4 inp00, inp10, inp20, inp30, inp40, inp50, inp60, inp70, inp80; 
      ae_int16x4 inp01, inp11, inp21, inp31, inp41, inp51, inp61, inp71, inp81; 
      ae_int16x4 inp02, inp12, inp22, inp32, inp42, inp52, inp62, inp72, inp82; 

      ae_int8x8 data00, data01;
      ae_int8x8 data10, data11;
      ae_int8x8 data20, data21;
      ae_int8x8 data30, data31;
      ae_int8x8 data40, data41;
      ae_int8x8 data50, data51;
      ae_int8x8 data60, data61;
      ae_int8x8 data70, data71;
      ae_int8x8 data80, data81;

      ae_int8x8 ker00, ker01; 
      ae_int8x8 ker10, ker11; 
      ae_int8x8 ker20, ker21; 

      /* Initialize accumulators with bias */
      ae_int32x2 temp0, temp1;
      AE_L32X2X2_IP(temp0, temp1, (ae_int32x4 *)p_accu, 16);
#if 0
      acc_ch0_01 = acc_ch0_23 = AE_SEL32_HH(temp0, temp0);
      acc_ch1_01 = acc_ch1_23 = AE_SEL32_LL(temp0, temp0);
      acc_ch2_01 = acc_ch2_23 = AE_SEL32_HH(temp1, temp1);
      acc_ch3_01 = acc_ch3_23 = AE_SEL32_LL(temp1, temp1);
#else
      ae_int8x8 ch0_01, ch0_23;
      ae_int8x8 ch1_01, ch1_23;
      ae_int8x8 ch2_01, ch2_23;
      ae_int8x8 ch3_01, ch3_23;

      AE_DSEL8X8(ch0_01, ch1_01, AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch0_23, ch1_23, AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(temp0), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch2_01, ch3_01, AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));
      AE_DSEL8X8(ch2_23, ch3_23, AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(temp1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32(0x73625140)));

      acc_ch0_01 = AE_MOVINT32X2_FROMINT8X8(ch0_01);
      acc_ch0_23 = AE_MOVINT32X2_FROMINT8X8(ch0_23);
      acc_ch1_01 = AE_MOVINT32X2_FROMINT8X8(ch1_01);
      acc_ch1_23 = AE_MOVINT32X2_FROMINT8X8(ch1_23);
      acc_ch2_01 = AE_MOVINT32X2_FROMINT8X8(ch2_01);
      acc_ch2_23 = AE_MOVINT32X2_FROMINT8X8(ch2_23);
      acc_ch3_01 = AE_MOVINT32X2_FROMINT8X8(ch3_01);
      acc_ch3_23 = AE_MOVINT32X2_FROMINT8X8(ch3_23);
#endif

      /* Load input */
      AE_L8X4S_XP(inp00, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp10, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp20, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp30, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp40, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp50, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp60, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp70, p_inp0, inp0_offset);
      AE_L8X4S_XP(inp80, p_inp0, - 8 *inp0_offset + 4);

      AE_L8X4S_XP(inp01, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp11, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp21, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp31, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp41, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp51, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp61, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp71, p_inp1, inp1_offset);
      AE_L8X4S_XP(inp81, p_inp1, -8 * inp1_offset + 4);

      AE_L8X4S_XP(inp02, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp12, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp22, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp32, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp42, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp52, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp62, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp72, p_inp2, inp2_offset);
      AE_L8X4S_XP(inp82, p_inp2, -8 * inp2_offset + 4);

      INTERLEAVE_3(data00, data01, inp00, inp01, inp02);
      INTERLEAVE_3(data10, data11, inp10, inp11, inp12);
      INTERLEAVE_3(data20, data21, inp20, inp21, inp22);
      INTERLEAVE_3(data30, data31, inp30, inp31, inp32);
      INTERLEAVE_3(data40, data41, inp40, inp41, inp42);
      INTERLEAVE_3(data50, data51, inp50, inp51, inp52);
      INTERLEAVE_3(data60, data61, inp60, inp61, inp62);
      INTERLEAVE_3(data70, data71, inp70, inp71, inp72);
      INTERLEAVE_3(data80, data81, inp80, inp81, inp82);

      /* Load kernel */
      AE_L8X8X2_IP(ker00, ker01, p_ker0, 16); // ch0, ch1, ch2, ch3
      AE_L8X8X2_IP(ker10, ker11, p_ker0, 16);
      AE_L8X8X2_IP(ker20, ker21, p_ker0, 16);

      /* Multiply and accumulate */
      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data00, data20, data40, data60, ker00); // ch0, ch1 row0
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data01, data21, data41, data61, ker01); // ch2, ch3 row0

      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data10, data30, data50, data70, ker10);
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data11, data31, data51, data71, ker11);

      AE_MULA4O8X8(acc_ch0_01, acc_ch0_23, acc_ch1_01, acc_ch1_23, data20, data40, data60, data80, ker20);
      AE_MULA4O8X8(acc_ch2_01, acc_ch2_23, acc_ch3_01, acc_ch3_23, data21, data41, data61, data81, ker21);

      /* Quantize */
      ae_int32x2 lmult01, lmult23;
      ae_int32x2 mult01, mult23;
      ae_int32x2 rmult01, rmult23;
      AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
      AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
      AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

#if 0
      d_acc01_0 = AE_SEL32_HH(acc_ch0_01, acc_ch1_01);
      d_acc23_0 = AE_SEL32_HH(acc_ch2_01, acc_ch3_01);
      d_acc01_1 = AE_SEL32_LL(acc_ch0_01, acc_ch1_01);
      d_acc23_1 = AE_SEL32_LL(acc_ch2_01, acc_ch3_01);
      d_acc01_2 = AE_SEL32_HH(acc_ch0_23, acc_ch1_23);
      d_acc23_2 = AE_SEL32_HH(acc_ch2_23, acc_ch3_23);
      d_acc01_3 = AE_SEL32_LL(acc_ch0_23, acc_ch1_23);
      d_acc23_3 = AE_SEL32_LL(acc_ch2_23, acc_ch3_23);
#else
      DSEL32X4_HHLL(d_acc01_0, d_acc01_1, acc_ch0_01, acc_ch1_01); 
      DSEL32X4_HHLL(d_acc23_0, d_acc23_1, acc_ch2_01, acc_ch3_01); 
      DSEL32X4_HHLL(d_acc01_2, d_acc01_3, acc_ch0_23, acc_ch1_23); 
      DSEL32X4_HHLL(d_acc23_2, d_acc23_3, acc_ch2_23, acc_ch3_23); 
#endif

      ae_int16x4 out_0, out_1, out_2, out_3;
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc01_0, d_acc23_0, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_1, d_acc01_1, d_acc23_1, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_2, d_acc01_2, d_acc23_2, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);
      MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_3, d_acc01_3, d_acc23_3, \
          mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

      AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
      AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

      /* Pack and store output */
      ae_int8x8 out32_0, out32_1;
      PACK_32X2(out32_0, out_0, out_1);
      PACK_32X2(out32_1, out_2, out_3);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_out0, input_channels * out_width);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_out0, -3 * input_channels * out_width + 4);
    }

    increment_input_pointer(p_state, 4 * y_stride);
    p_out += 4 * input_channels * out_width;
  }
  return p_out;
}

static inline void process_three_input_rows
  (pWORD8 __restrict__ p_out
  ,WORD32  input_channels
  ,WORD32  out_zero_bias
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_ch;
  const WORD8 * __restrict__ p_inp0, *__restrict__ p_inp1, *__restrict__ p_inp2;
  const WORD8 * __restrict__ p_ker0;
  int inp0_offset, inp1_offset, inp2_offset;

  const WORD32 * __restrict__ p_scale_multipliers = p_state->p_scale_multipliers;

  const WORD8 *__restrict__ p_kernel = p_state->p_kernel_rearranged;
  const WORD32 *__restrict__ p_accu = p_state->p_accu_zero_point;

  /* Set up input and kernel pointers */
  p_inp0 = p_state->p_inp0;
  p_inp1 = p_state->p_inp1;
  p_inp2 = p_state->p_inp2;

  inp0_offset = p_state->inp0_offset;
  inp1_offset = p_state->inp1_offset;
  inp2_offset = p_state->inp2_offset;

  p_ker0 = p_kernel;

  for(itr_ch = 0; itr_ch < input_channels; itr_ch += 4)
  {
    ae_int32x2 d_acc23, d_acc10;

    ae_int16x4 inp00, inp01, inp02; 
    ae_int16x4 inp10, inp11, inp12; 
    ae_int16x4 inp20, inp21, inp22; 

    ae_int16x4 ker00, ker01, ker02; 
    ae_int16x4 ker10, ker11, ker12; 
    ae_int16x4 ker20, ker21, ker22; 

    /* Initialize accumulators with bias */
    AE_L32X2X2_IP(d_acc23, d_acc10, (ae_int32x4 *)p_accu, 16);

    /* Load input */
    inp20 = AE_L8X4S_X(p_inp0, 2 * inp0_offset);
    inp10 = AE_L8X4S_X(p_inp0, inp0_offset);
    AE_L8X4S_IP(inp00, p_inp0, 4);
    inp21 = AE_L8X4S_X(p_inp1, 2 * inp1_offset);
    inp11 = AE_L8X4S_X(p_inp1, inp1_offset);
    AE_L8X4S_IP(inp01, p_inp1, 4);
    inp22 = AE_L8X4S_X(p_inp2, 2 * inp2_offset);
    inp12 = AE_L8X4S_X(p_inp2, inp2_offset);
    AE_L8X4S_IP(inp02, p_inp2, 4);

    /* Load kernel */
    AE_L8X4S_IP(ker00, p_ker0, 4);
    AE_L8X4S_IP(ker01, p_ker0, 4);
    AE_L8X4S_IP(ker02, p_ker0, 4);
    AE_L8X4S_IP(ker10, p_ker0, 4);
    AE_L8X4S_IP(ker11, p_ker0, 4);
    AE_L8X4S_IP(ker12, p_ker0, 4);
    AE_L8X4S_IP(ker20, p_ker0, 4);
    AE_L8X4S_IP(ker21, p_ker0, 4);
    AE_L8X4S_IP(ker22, p_ker0, 4);

    /* Multiply and accumulate */
    AE_MULA16X4(d_acc23, d_acc10, inp00, ker00);
    AE_MULA16X4(d_acc23, d_acc10, inp01, ker01);
    AE_MULA16X4(d_acc23, d_acc10, inp02, ker02);
    AE_MULA16X4(d_acc23, d_acc10, inp10, ker10);
    AE_MULA16X4(d_acc23, d_acc10, inp11, ker11);
    AE_MULA16X4(d_acc23, d_acc10, inp12, ker12);
    AE_MULA16X4(d_acc23, d_acc10, inp20, ker20);
    AE_MULA16X4(d_acc23, d_acc10, inp21, ker21);
    AE_MULA16X4(d_acc23, d_acc10, inp22, ker22);

    /* Quantize */
    ae_int32x2 lmult01, lmult23;
    ae_int32x2 mult01, mult23;
    ae_int32x2 rmult01, rmult23;
    AE_L32X2X2_IP(lmult01, lmult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(mult01, mult23, (ae_int32x4*)p_scale_multipliers, 16);
    AE_L32X2X2_IP(rmult01, rmult23, (ae_int32x4*)p_scale_multipliers, 16);

    ae_int16x4 out_0;
    MPY_BY_QUANT_MULT_PER_CHAN_LR_MULT_X2X2_OUT16_ZB(out_0, d_acc23, d_acc10, \
        mult01, mult23, lmult01, lmult23, rmult01, rmult23, out_zero_bias);

    AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));

    /* Pack and store output */
    ae_int8x8 out32_0;
    PACK_32X2(out32_0, out_0, AE_ZERO16());
    AE_S32_H_IP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_out, 4);
  }
}


static void assign_input_pointers_and_offsets
    (xa_nn_conv2d_dw_k3x3_state_t *p_state
    ,const WORD8 * p_inp 
    ,WORD32 x_start
    ,WORD32 x_padding
    ,WORD32 input_width
    ,WORD32 input_channels
    )
{
  /* Set up input and kernel pointers */
  if(x_start < x_padding)
  {
    p_state->p_inp0 = p_state->p_dummy_inp;
    p_state->inp0_offset = 0;
  }
  else
  {
    p_state->p_inp0 = p_inp;
    p_state->inp0_offset = input_width * input_channels;
  }

  if((x_start + 1 < x_padding) || (x_start + 1 >= (x_padding + input_width)))
  {
    p_state->p_inp1 = p_state->p_dummy_inp;
    p_state->inp1_offset = 0;
  }
  else
  {
    p_state->p_inp1 = (p_inp + input_channels);
    p_state->inp1_offset = input_width * input_channels;
  }

  if((x_start + 2 >= (x_padding + input_width)))
  {
    p_state->p_inp2 = p_state->p_dummy_inp;
    p_state->inp2_offset = 0;
  }
  else
  {
    p_state->p_inp2 = (p_inp + 2 * input_channels);
    p_state->inp2_offset = input_width * input_channels;
  }
}

static void process_single_output_vplane_single_stride
  (pWORD8 __restrict__ p_out
  ,const WORD32 * p_bias
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,WORD32 input_height
  ,WORD32 input_width
  ,WORD32 input_channels
  ,WORD32 y_stride
  ,WORD32 x_padding
  ,WORD32 y_padding
  ,WORD32 out_height
  ,WORD32 out_width
  ,WORD32 input_zero_bias
  ,WORD32 out_zero_bias
  ,WORD32 x_start
  ,xa_nn_conv2d_dw_k3x3_state_t *p_state
  )
{
  int itr_oh;

  for(itr_oh = 0; itr_oh < p_state->top_padded_region_output; itr_oh++)
  {
    process_padded_region_output_row(p_out
        ,p_bias
        ,input_channels
        ,out_zero_bias
        ,p_state
        );
    p_out += input_channels * out_width;
  }
  if(p_state->top_single_input_row_output)
  {
    int kernel_height_offset;
    kernel_height_offset = 2; // bottom row 

    process_single_input_row(p_out
        ,p_kernel + kernel_height_offset * KW_3X3 * input_channels
        ,input_channels
        ,input_zero_bias
        ,out_zero_bias
        ,p_state
        );
    p_out += input_channels * out_width;
  }
  if(p_state->middle_single_input_row_output)
  {
    int kernel_height_offset;
    kernel_height_offset = 1; //middle row

    process_single_input_row(p_out
        ,p_kernel + kernel_height_offset * KW_3X3 * input_channels
        ,input_channels
        ,input_zero_bias
        ,out_zero_bias
        ,p_state
        );
    p_out += input_channels * out_width;
  }

  if(p_state->top_two_input_row_output)
  {
    int kernel_height_offset;
    kernel_height_offset = 1;

    process_two_input_rows(p_out
        ,p_kernel + kernel_height_offset * KW_3X3 * input_channels
        ,input_channels
        ,input_zero_bias
        ,out_zero_bias
        ,p_state
        );

    p_out += input_channels * out_width;
    if(y_stride == 2)
      increment_input_pointer(p_state, 1);
  }

  if(y_stride == 1)
  {
    p_out = process_six_input_rows(p_out
              ,input_channels
              ,out_width
              ,out_zero_bias
              ,p_state
              );
  }
  else
  {
    p_out = process_six_input_rows_ystride(p_out
              ,input_channels
              ,y_stride
              ,out_width
              ,out_zero_bias
              ,p_state
              );
  }

  for(itr_oh = 0; itr_oh < p_state->three_input_row_output; itr_oh++)
  {
    process_three_input_rows(p_out
        ,input_channels
        ,out_zero_bias
        ,p_state
        );
    increment_input_pointer(p_state, y_stride);
    p_out += input_channels * out_width;
  }

  if(p_state->bottom_two_input_row_output)
  {
    int kernel_height_offset;
    kernel_height_offset = 0;

    process_two_input_rows(p_out
        ,p_kernel + kernel_height_offset * KW_3X3 * input_channels
        ,input_channels
        ,input_zero_bias
        ,out_zero_bias
        ,p_state
        );

    increment_input_pointer(p_state, y_stride);
    p_out += input_channels * out_width;
  }

  if(p_state->bottom_single_input_row_output)
  {
    int kernel_height_offset;
    kernel_height_offset = 0; //top row 

    process_single_input_row(p_out
        ,p_kernel + kernel_height_offset * KW_3X3 * input_channels
        ,input_channels
        ,input_zero_bias
        ,out_zero_bias
        ,p_state
        );
    p_out += input_channels * out_width;
  }
  for(itr_oh = 0; itr_oh < p_state->bottom_padded_region_output; itr_oh++)
  {
    process_padded_region_output_row(p_out
        ,p_bias
        ,input_channels
        ,out_zero_bias
        ,p_state
        );
    p_out += input_channels * out_width;
  }
}
#endif

/* Special case of 3x3 kernel for NHWC format
   Supports multiple of 4 channels, y_stride 1  
   Channel multiplier should be 1
 */
WORD32 xa_nn_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s_k3x3
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  int i;
  //TODO: input pointer alignment check
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT_16, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0) || ((input_channels & 0x3) != 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height != 3) || (kernel_width != 3), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier != 1), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias > 128 || input_zero_bias < -127), -1);
  for(i = 0; i < input_channels*channels_multiplier; i++)
    XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 31), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
  /* Implementation dependent checks */
  //TOOD: support y_stride 2
  XA_NNLIB_ARG_CHK_COND((y_stride != 1) && (y_stride != 2), -1);

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
  WORD32 input_zero_bias_neg = -input_zero_bias;
  int itr_ow;
  xa_nn_conv2d_dw_k3x3_state_t * p_state;

  p_state = xa_nn_conv2d_depthwise_init_nhwc_k3x3
    (p_scratch
    ,p_kernel
    ,p_bias
    ,input_height
    ,input_channels
    ,out_height
    ,y_stride
    ,y_padding
    ,input_zero_bias_neg
    ,p_out_multiplier
    ,p_out_shift
    );

  /* Process one output vertical plane at a time, incase later we use
     circular buffer */
#pragma loop_count min=1
  for(itr_ow = 0; itr_ow < out_width; itr_ow++)
  {
    int x_start = itr_ow * x_stride;
    int x_stop = x_start + kernel_width - 1;

    if((x_stop < x_padding) || (x_start >= (x_padding + input_width)))
    {
      process_padded_region_output_vplane(p_out + itr_ow * input_channels
          ,p_bias
          ,out_height
          ,out_width
          ,input_channels
          ,out_zero_bias
          ,p_state
          );
    }
    else
    {
      assign_input_pointers_and_offsets(p_state
          ,p_inp + (x_start - x_padding) * input_channels
          ,x_start
          ,x_padding
          ,input_width
          ,input_channels
          );

      //TODO: y_stride == 1 case and y_stride >= 1 case
      process_single_output_vplane_single_stride
        (p_out + itr_ow * input_channels
        ,p_bias
        ,p_kernel
        ,p_inp + (x_start - x_padding) * input_channels
        ,input_height
        ,input_width
        ,input_channels
        ,y_stride
        ,x_padding
        ,y_padding
        ,out_height
        ,out_width
        ,input_zero_bias_neg
        ,out_zero_bias
        ,x_start 
        ,p_state
        );
    }
  }
#else
  xa_nn_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s
    (p_out
    ,p_kernel
    ,p_inp
    ,p_bias
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,x_stride
    ,y_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,out_width
    ,input_zero_bias
    ,p_out_multiplier
    ,p_out_shift
    ,out_zero_bias
    ,out_data_format
    ,p_scratch
    );
#endif

  return 0;
}

WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s_generic
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  int i;
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT_16, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias > 128 || input_zero_bias < -127), -1);
  for(i = 0; i < input_channels*channels_multiplier; i++)
    XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 31), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  if(inp_data_format == 0)
  {
    xa_nn_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s
      (p_out
      ,p_kernel
      ,p_inp
      ,p_bias
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,y_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,out_width
      ,input_zero_bias
      ,p_out_multiplier
      ,p_out_shift
      ,out_zero_bias
      ,out_data_format
      ,p_scratch
      );
  }
  else if(inp_data_format == 1)
  {
    xa_nn_conv2d_depthwise_nchw_per_chan_sym8sxasym8s
      (p_out
      ,p_kernel
      ,p_inp
      ,p_bias
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,y_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,out_width
      ,input_zero_bias
      ,p_out_multiplier
      ,p_out_shift
      ,out_zero_bias
      ,out_data_format
      ,p_scratch
      );
  }
  return 0;
}

static void xa_nn_rearrange_hwc_to_chw
              (pWORD8 __restrict__ p_out
              ,const WORD8*  __restrict__ p_inp
              ,WORD32 height
              ,WORD32 width
              ,WORD32 channels
              ) 
{
  int itr_ch, itr_h, itr_w;
  for(itr_ch = 0; itr_ch < channels; itr_ch++)
  {
    for(itr_h = 0; itr_h < height; itr_h++)
    {
      for(itr_w = 0; itr_w < width; itr_w++)
      {
        p_out[itr_ch * height * width + itr_h * width + itr_w] =
          p_inp[itr_h * width * channels + itr_w * channels + itr_ch];
      }
    }
  }
}

WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD32 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  /* For single input channel, use the standard convolution */
  if((input_channels == 1) && 
     (inp_data_format == 0)
    )
  {
    pWORD8 p_kernel_nchw;
    p_scratch = (void *)ALIGN_PTR(p_scratch, ALIGNMENT_16);
    p_kernel_nchw = (pWORD8)p_scratch;
    p_scratch += ALIGNED_SIZE(channels_multiplier * kernel_height * kernel_width, ALIGNMENT_16);

    /* Rearrange the kernel in NCHW format */
    xa_nn_rearrange_hwc_to_chw(p_kernel_nchw, p_kernel, kernel_height, kernel_width, channels_multiplier);
    
    return xa_nn_conv2d_std_per_chan_sym8sxasym8s
      (p_out
      ,p_inp
      ,p_kernel_nchw
      ,p_bias
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,y_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,out_width
      ,input_zero_bias
      ,(WORD32 *)p_out_multiplier
      ,(WORD32 *)p_out_shift
      ,out_zero_bias
      ,out_data_format
      ,p_scratch
      );
  }
#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
  else if((channels_multiplier == 1) &&
     (kernel_height == 3) &&
     (kernel_width == 3) &&
      ALIGNED_PTR(p_inp, 4) &&
      ALIGNED_PTR(p_kernel, 4) &&
      ALIGNED_PTR(p_out, 4) &&
     ((y_stride == 1) || (y_stride == 2)) &&
     (inp_data_format == 0) &&
     ((input_channels & 0x3) == 0) &&
     1)
  {
    return xa_nn_conv2d_depthwise_nhwc_per_chan_sym8sxasym8s_k3x3
      (p_out
      ,p_kernel
      ,p_inp
      ,p_bias
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,y_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,out_width
      ,input_zero_bias
      ,p_out_multiplier
      ,p_out_shift
      ,out_zero_bias
      ,inp_data_format
      ,out_data_format
      ,p_scratch
      );
  }
#endif
  else 
  {
    return xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s_generic
      (p_out
      ,p_kernel
      ,p_inp
      ,p_bias
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,y_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,out_width
      ,input_zero_bias
      ,p_out_multiplier
      ,p_out_shift
      ,out_zero_bias
      ,inp_data_format
      ,out_data_format
      ,p_scratch
      );
  }
}
