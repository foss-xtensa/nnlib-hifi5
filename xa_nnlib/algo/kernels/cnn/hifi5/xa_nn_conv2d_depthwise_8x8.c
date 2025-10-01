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
#include <string.h>
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

/* 2D Convolution implementation */
static inline void conv2d_nchw_8x8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_ker
  ,const WORD8 *__restrict__ p_inp
  ,WORD8 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height
  ,int actual_out_width
  ,int out_stride
  ,int x_stride
  ,int y_stride
  ,WORD32  acc_shift
  ,WORD32  bias_shift
  ,pWORD32 __restrict__ p_scratch
  )
{
  (VOID) input_height;
  int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
  int kernel_width_pad = (kernel_width + 3) & (~3);

  /* Generic case */
  int i, j, k, l;
  int output_width_for_x_stride_1;

  /* Here input_width is nothing but circ_buf_width, which is taken care to be
   * multiple of 8. */
  output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
  /* output_width_for_x_stride_1 loop is unrolled by 8 */
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, 8);

  ae_int32x2 d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3;
  ae_int32 *scratch_ptr = (ae_int32 *)p_scratch;

  ae_int64 d_bias;
  d_bias = AE_MOVINT64_FROMF64(AE_CVT64A32(bias));
  d_bias = SW_SLAA64S_INT64_INT64(d_bias, bias_shift - 32);

  if(kernel_width_pad==12)
  {
    ae_int8x8 d_inp00, d_inp01, d_inp02;
    ae_int8x8 d_inp10, d_inp11, d_inp12;
    ae_int8x8 d_ker0, d_ker1, d_ker2;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j += 8)
      {
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
        ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)(p_inp);
        AE_ADDCIRC16X4_XC(pt16x4_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)pt16x4_inp0;
        ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
        for(k = 0; k < (kernel_height_pad >> 1); k++)
        {
          AE_L8X8_IP(d_ker0, pt_ker, 8);
          AE_L8X8_IP(d_ker1, pt_ker, 8);
          AE_L8X8_IP(d_ker2, pt_ker, 8);

          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, 8);
          AE_L8X8_XC(d_inp02, pt_inp0, (input_width - 16));
          AE_L8X8_XC(d_inp10, pt_inp0, 8);
          AE_L8X8_XC(d_inp11, pt_inp0, 8);
          AE_L8X8_XC(d_inp12, pt_inp0, (input_width - 16));

          AE_MULA8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          AE_MULA8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          AE_MULA4O8X8CNV_H(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker1, d_inp01, d_inp02);
          AE_MULA4O8X8CNV_L(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
          AE_MULA8Q8X8CNV_L(d_acc32_0, d_acc32_1, d_ker2, d_inp10, d_inp11);
          AE_MULA8Q8X8CNV_H(d_acc32_2, d_acc32_3, d_ker2, d_inp11, d_inp12);
        }
        ae_int32x4 *p_sc = (ae_int32x4 *)(scratch_ptr + j);
        AE_S32X2X2_I(d_acc32_0, d_acc32_1, p_sc,  0);
        AE_S32X2X2_I(d_acc32_2, d_acc32_3, p_sc, 16);
      }
    }
  }
  else if(kernel_width_pad==8)
  {
    ae_int8x8 d_inp00, d_inp01;
    ae_int8x8 d_inp10, d_inp11;
    ae_int8x8 d_ker0;
    ae_int8x8 d_ker1;
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
#pragma loop_count min=1
      for(j = 0; j < output_width_for_x_stride_1; j += 8)
      {
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
        ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)(p_inp);
        AE_ADDCIRC16X4_XC(pt16x4_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)pt16x4_inp0;
        ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
#pragma no_unroll
        for(k = 0; k < (kernel_height_pad >> 1); k++)
        {
          AE_L8X8_IP(d_ker0, pt_ker, 8);
          AE_L8X8_IP(d_ker1, pt_ker, 8);
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, (input_width - 8));
          AE_L8X8_XC(d_inp10, pt_inp0, 8);
          AE_L8X8_XC(d_inp11, pt_inp0, (input_width - 8));

          AE_MULA8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          AE_MULA8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          AE_MULA8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
          AE_MULA8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
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
      for(j = 0; j < output_width_for_x_stride_1; j += 16)
      {
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
        d_acc32_4 = AE_ZERO32();
        d_acc32_5 = AE_ZERO32();
        d_acc32_6 = AE_ZERO32();
        d_acc32_7 = AE_ZERO32();

        ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)(p_inp);
        ae_int16x4 *pt16x4_inp1 = (ae_int16x4 *)(p_inp);
        AE_ADDCIRC16X4_XC(pt16x4_inp0, sizeof(WORD8) * (((i * y_stride) * input_width) + j));
        AE_ADDCIRC16X4_XC(pt16x4_inp1, sizeof(WORD8) * (((i * y_stride + 1) * input_width) + j));
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)pt16x4_inp0;
        ae_int8x8 *pt_inp1 = (ae_int8x8 *)pt16x4_inp1;

        ae_int8x8 *pt_ker0 = (ae_int8x8 *)(p_ker);

#pragma no_unroll
#pragma loop_count min=1
        for(k = 0; k < (kernel_height_pad >> 1); k++)
        {
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp01, pt_inp0, 8);
          AE_L8X8_XC(d_inp02, pt_inp0, (2*input_width - 16));
          AE_L8X8_XC(d_inp10, pt_inp1, 8);
          AE_L8X8_XC(d_inp11, pt_inp1, 8);
          AE_L8X8_XC(d_inp12, pt_inp1, (2*input_width - 16));
          AE_L8X8_IP(d_ker0, pt_ker0, 8);

          AE_MULA2X4Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp10);
          AE_MULA2X4Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01, d_inp10, d_inp11);
          AE_MULA2X4Q8X8CNV_H(d_acc32_4, d_acc32_5, d_ker0, d_inp01, d_inp11);
          AE_MULA2X4Q8X8CNV_L(d_acc32_6, d_acc32_7, d_ker0, d_inp01, d_inp02, d_inp11, d_inp12);
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
      for(j = 0; j < output_width_for_x_stride_1; j += 8)
      {
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
#pragma loop_count min=1
        for(k = 0; k < kernel_height_pad; k += 2)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)(p_inp);
          AE_ADDCIRC16X4_XC(pt16x4_inp, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j));
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)pt16x4_inp;

          pt16x4_inp = (ae_int16x4 *)(p_inp);
          AE_ADDCIRC16X4_XC(pt16x4_inp, sizeof(WORD8) * (((i * y_stride + k + 1) * input_width) + j));
          ae_int8x8 *pt_inp1 = (ae_int8x8 *)pt16x4_inp;
          /* Start of scratch memory for padded kernel is 8-byte aligned and
           * and kernel width is padded to be multiple of 4, so every alternate
           * row is 8-byte aligned for kernel */
          ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[k * kernel_width_pad]);
          ae_int8x8 *pt_ker1 = (ae_int8x8 *)(&p_ker[(k + 1) * kernel_width_pad]);
          ae_valign ker_a = AE_LA64_PP(pt_ker1);
          AE_L8X8_XC(d_inp00, pt_inp0, 8);
          AE_L8X8_XC(d_inp10, pt_inp1, 8);
#pragma no_unroll
#pragma loop_count min=1
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_L8X8_XC(d_inp01, pt_inp0, 8);
            AE_L8X8_XC(d_inp11, pt_inp1, 8);
            AE_L8X8_IP(d_ker0, pt_ker0, 8);
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker1);
            AE_MULA8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
            AE_MULA8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
            AE_MULA8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
            AE_MULA8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
            d_inp00 = d_inp01;
            d_inp10 = d_inp11;
          }
        }
        if(kernel_width_pad & 4)
        {
#pragma loop_count min=1
          for(k = 0; k < kernel_height_pad; k += 2)
          {
            ae_int16x4 *pt16x4_inp = (ae_int16x4 *)(p_inp);
            AE_ADDCIRC16X4_XC(pt16x4_inp, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j + (kernel_width_pad & (~7))));
            ae_int8x8 *pt_inp0 = (ae_int8x8 *)pt16x4_inp;
            pt16x4_inp = (ae_int16x4 *)(p_inp);
            AE_ADDCIRC16X4_XC(pt16x4_inp, sizeof(WORD8) * (((i * y_stride + k + 1) * input_width) + j + (kernel_width_pad & (~7))));
            ae_int8x8 *pt_inp1 = (ae_int8x8 *)pt16x4_inp;
            /* Start of scratch memory for padded kernel is 8-byte aligned and
             * and kernel width is padded to be multiple of 4, so every
             * alternate row is 8-byte aligned for kernel */
            ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[(k + 1) * kernel_width_pad - 4]);
            ae_int8x8 *pt_ker1 = (ae_int8x8 *)(&p_ker[(k + 2) * kernel_width_pad - 8]);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp10, pt_inp1, 8);
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_L8X8_XC(d_inp11, pt_inp1, 8);
              AE_L8X8_IP(d_ker0, pt_ker0, 8);
              AE_L8X8_IP(d_ker1, pt_ker1, 8);
              AE_MULA4O8X8CNV_H(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
              AE_MULA4O8X8CNV_L(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
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
   * will be picked from it as per 'x_stride'. y_stride is already taken care */
  ae_int32 *scratch_ptr1 = (ae_int32 *) p_scratch;

  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr1 = (ae_int32 *) p_scratch + (i * output_width_for_x_stride_1);
    WORD8 *out_ptr  = (WORD8 *) p_out + (i * out_stride * actual_out_width);
    ae_int32x2 accu_int32_0;
    ae_int64 accu_int64_0, accu_int64_1;
    ae_int8x8 accu_int8x8;

    for(j = 0; j < actual_out_width; j++)
    {
      accu_int32_0 = AE_MOVINT32X2_FROMINT32(scratch_ptr1[(j * x_stride)]);
      AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32_0, AE_ZERO32());
      accu_int64_0 = SW_ADD64S_INT64_INT64(accu_int64_0, d_bias);
      accu_int64_0 = SW_SLAA64S_INT64_INT64(accu_int64_0, acc_shift + 32);
      accu_int32_0 = AE_MOVINT32X2_FROMF32X2(AE_ROUND32F64SSYM(AE_MOVF64_FROMINT64(accu_int64_0)));
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

static void xa_nn_conv2d_depthwise_nchw_8x8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD8 *__restrict__ p_bias
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
  ,WORD32  acc_shift
  ,WORD32  bias_shift
  ,pVOID p_scratch
  )
{
  WORD8 pad_val = 0;
  xa_nn_dilated_conv2d_depthwise_init
    (p_scratch
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,1
    ,1
    ,x_stride
    ,y_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,out_width
    ,8
    ,1
    ,(pVOID)(&pad_val)
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

    CIRC_BUF_ADD_ROWS_INIT
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
      );

#pragma loop_count min=1
    for(itr_oh = 0; itr_oh < out_height; itr_oh += circ_out_height)
    {
      CIRC_BUF_ADD_ROWS
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
        );

      p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

#pragma loop_count min=1
      for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
      {
        pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
        COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
        bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

        conv2d_nchw_8x8
          ((WORD8 *)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
          ,p_kernel_padded
          ,p_inp_circ
          ,(WORD8)bias
          ,p_circ_buf->rows
          ,p_circ_buf->row_offset
          ,kernel_height
          ,kernel_width
          ,XT_MIN(circ_out_height, out_height-itr_oh)
          ,out_width
          ,(input_channels * channels_multiplier)
          ,x_stride
          ,y_stride
          ,acc_shift
          ,bias_shift
          ,p_tmp_out
          );
      }
    }
  }
}

#define LOAD_SHIFT_BIAS_TO64(d64, bias_val, shift) \
d64 = AE_MOVINT64_FROMF64(AE_CVT64A32(bias_val)); \
d64 = SW_SLAA64S_INT64_INT64(d64, shift - 32);

/* 2D Convolution implementation */
static inline void conv2d_nhwc_8x8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_ker
  ,const WORD8 *__restrict__ p_inp
  ,const WORD8 *p_bias
  ,int kernel_height
  ,int kernel_width
  ,int out_height
  ,int out_width
  ,int out_channels
  ,int x_stride
  ,int y_stride
  ,WORD32  acc_shift
  ,WORD32  bias_shift
  ,pWORD32 __restrict__ p_scratch
  )
{
  (VOID) x_stride;
  (VOID) p_scratch;
  WORD32 ker_channels_pad, inp_channels_pad;
  WORD32 i, itr_oh, itr_ch, itr_kw;
  ae_int8x8 *pt_inp0;
  pWORD8 pt_ker;
  pWORD8 out_ptr0, out_ptr1;
  ae_int8x8 *ae_out_ptr0, *ae_out_ptr1;
  ae_valign out0_a, out1_a;
  ae_int16x4 d_inp00, d_inp01;
  ae_int16x4 d_inp10, d_inp11;
  const WORD8 *pt_bias;
  ae_int32x2 d_acc0, d_acc1, d_acc2, d_acc3;
  ae_int64 d_acc64_0, d_acc64_1, d_acc64_2, d_acc64_3;
  ae_int64 d_acc64_4, d_acc64_5, d_acc64_6, d_acc64_7;
  ae_int64 d_bias64_0, d_bias64_1, d_bias64_2, d_bias64_3;
  ae_int64 d_bias64_4, d_bias64_5, d_bias64_6, d_bias64_7;
  ae_int32x2 d_acc4, d_acc5, d_acc6, d_acc7;
  ae_int8x8 d_acc8x8;

  ker_channels_pad = out_channels;
#ifndef AE_MULZB3X3O8X8
  inp_channels_pad = (out_channels + 7)&(~7);
#else
  inp_channels_pad = (out_channels + 15)&(~15);
#endif

  /* For acc_shift on accumulator */
  WUR_AE_SAR(acc_shift + 32);
#pragma loop_count min=1
  for(itr_oh = 0; itr_oh < (out_height); itr_oh += 2)
  {
    out_ptr0 = (WORD8 *)(&p_out[itr_oh * out_channels * out_width]);
    out_ptr1 = (WORD8 *)(&p_out[(itr_oh + 1) * out_channels * out_width]);
    ae_out_ptr0 = (ae_int8x8 *)(out_ptr0);
    ae_out_ptr1 = (ae_int8x8 *)(out_ptr1);
    out0_a = AE_ZALIGN64();
    out1_a = AE_ZALIGN64();
    pt_bias = (const WORD8 *)p_bias;
#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < out_channels; itr_ch += 8)
    {
      ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
      AE_ADDCIRC16X4_XC(pt16x4_inp0, itr_ch + itr_oh * y_stride * kernel_width * inp_channels_pad);
      pt_inp0 = (ae_int8x8 *)pt16x4_inp0;
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
      ae_int8x8 d_z8x8 = AE_MOVINT8X8_FROMINT16X4(AE_ZERO16());
#pragma no_unroll
#pragma loop_count min=1
      for(itr_kw = 0; itr_kw < kernel_height * kernel_width; itr_kw++)
      {
        AE_L8X8_XC(d_inp0, pt_inp0, y_stride * kernel_width * inp_channels_pad);
        AE_L8X8_XC(d_inp1, pt_inp0, inp_channels_pad - y_stride * kernel_width * inp_channels_pad);
        AE_LA8X8_IP(d_ker, ker_a, ptae_ker);
        ptae_ker = (ae_int8x8 *)((WORD8 *)ptae_ker + (ker_channels_pad - 8));
        ker_a = AE_LA64_PP(ptae_ker);
        AE_ADDW8(d_inp00, d_inp01, d_inp0, d_z8x8);
        AE_ADDW8(d_inp10, d_inp11, d_inp1, d_z8x8);
        AE_ADDW8(d_ker0, d_ker1, d_ker, d_z8x8);
        AE_MULA16X4(d_acc0, d_acc1, d_inp00, d_ker0);
        AE_MULA16X4(d_acc2, d_acc3, d_inp01, d_ker1);
        AE_MULA16X4(d_acc4, d_acc5, d_inp10, d_ker0);
        AE_MULA16X4(d_acc6, d_acc7, d_inp11, d_ker1);
      }
      LOAD_SHIFT_BIAS_TO64(d_bias64_0, pt_bias[itr_ch + 0], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_1, pt_bias[itr_ch + 1], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_2, pt_bias[itr_ch + 2], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_3, pt_bias[itr_ch + 3], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_4, pt_bias[itr_ch + 4], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_5, pt_bias[itr_ch + 5], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_6, pt_bias[itr_ch + 6], bias_shift);
      LOAD_SHIFT_BIAS_TO64(d_bias64_7, pt_bias[itr_ch + 7], bias_shift);

      AE_ADDW32(d_acc64_0, d_acc64_1, d_acc0, AE_ZERO32());
      AE_ADDW32(d_acc64_2, d_acc64_3, d_acc1, AE_ZERO32());
      AE_ADDW32(d_acc64_4, d_acc64_5, d_acc2, AE_ZERO32());
      AE_ADDW32(d_acc64_6, d_acc64_7, d_acc3, AE_ZERO32());
      d_acc64_0 = SW_ADD64S_INT64_INT64(d_acc64_0, d_bias64_0);
      d_acc64_1 = SW_ADD64S_INT64_INT64(d_acc64_1, d_bias64_1);
      d_acc64_2 = SW_ADD64S_INT64_INT64(d_acc64_2, d_bias64_2);
      d_acc64_3 = SW_ADD64S_INT64_INT64(d_acc64_3, d_bias64_3);
      d_acc64_4 = SW_ADD64S_INT64_INT64(d_acc64_4, d_bias64_4);
      d_acc64_5 = SW_ADD64S_INT64_INT64(d_acc64_5, d_bias64_5);
      d_acc64_6 = SW_ADD64S_INT64_INT64(d_acc64_6, d_bias64_6);
      d_acc64_7 = SW_ADD64S_INT64_INT64(d_acc64_7, d_bias64_7);

      d_acc64_0 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_0)));
      d_acc64_1 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_1)));
      d_acc64_2 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_2)));
      d_acc64_3 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_3)));
      d_acc64_4 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_4)));
      d_acc64_5 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_5)));
      d_acc64_6 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_6)));
      d_acc64_7 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_7)));

      d_acc0 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_0, d_acc64_1);
      d_acc1 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_2, d_acc64_3);
      d_acc2 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_4, d_acc64_5);
      d_acc3 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_6, d_acc64_7);

      d_acc8x8 = AE_SEL8X8I(AE_SAT8X4X32_L(d_acc0, d_acc1), AE_SAT8X4X32_L(d_acc2, d_acc3), 3);
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

      AE_ADDW32(d_acc64_0, d_acc64_1, d_acc4, AE_ZERO32());
      AE_ADDW32(d_acc64_2, d_acc64_3, d_acc5, AE_ZERO32());
      AE_ADDW32(d_acc64_4, d_acc64_5, d_acc6, AE_ZERO32());
      AE_ADDW32(d_acc64_6, d_acc64_7, d_acc7, AE_ZERO32());
      d_acc64_0 = SW_ADD64S_INT64_INT64(d_acc64_0, d_bias64_0);
      d_acc64_1 = SW_ADD64S_INT64_INT64(d_acc64_1, d_bias64_1);
      d_acc64_2 = SW_ADD64S_INT64_INT64(d_acc64_2, d_bias64_2);
      d_acc64_3 = SW_ADD64S_INT64_INT64(d_acc64_3, d_bias64_3);
      d_acc64_4 = SW_ADD64S_INT64_INT64(d_acc64_4, d_bias64_4);
      d_acc64_5 = SW_ADD64S_INT64_INT64(d_acc64_5, d_bias64_5);
      d_acc64_6 = SW_ADD64S_INT64_INT64(d_acc64_6, d_bias64_6);
      d_acc64_7 = SW_ADD64S_INT64_INT64(d_acc64_7, d_bias64_7);

      d_acc64_0 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_0)));
      d_acc64_1 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_1)));
      d_acc64_2 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_2)));
      d_acc64_3 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_3)));
      d_acc64_4 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_4)));
      d_acc64_5 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_5)));
      d_acc64_6 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_6)));
      d_acc64_7 = AE_MOVINT64_FROMF64(AE_SLAS64S(AE_MOVF64_FROMINT64(d_acc64_7)));

      d_acc4 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_0, d_acc64_1);
      d_acc5 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_2, d_acc64_3);
      d_acc6 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_4, d_acc64_5);
      d_acc7 = SW_ROUND32X2F64SSYM_INT64_INT64_INT32X2(d_acc64_6, d_acc64_7);

      d_acc8x8 = AE_SEL8X8I(AE_SAT8X4X32_L(d_acc4, d_acc5), AE_SAT8X4X32_L(d_acc6, d_acc7), 3);
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

static void xa_nn_conv2d_depthwise_nhwc_8x8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD8 *__restrict__ p_bias
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
  ,WORD32  acc_shift
  ,WORD32  bias_shift
  ,pVOID p_scratch
  )
{
  WORD8 pad_val = 0;
  xa_nn_dilated_conv2d_depthwise_init
    (p_scratch
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,1
    ,1
    ,x_stride
    ,y_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,out_width
    ,8
    ,0
    ,(pVOID)(&pad_val)
    );

  xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = p_state;
  int itr_ow;
  int cols_to_add, left_pad, right_pad, cols_added;
  int input_col;
  const WORD8 *pt_inp;
  pWORD8 p_inp_circ;

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  pt_inp = (const WORD8 *)p_inp;

  CIRC_BUF_ADD_COLS_INIT
    (cols_added
    ,cols_to_add
    ,left_pad
    ,right_pad
    ,input_col
    ,input_height
    ,input_width
    ,input_channels
    ,kernel_height
    ,kernel_width
    ,channels_multiplier
    ,x_stride
    ,x_padding
    ,y_padding
    ,out_height
    ,p_circ_buf
    ,pt_inp
    );

#pragma loop_count min=1
  for(itr_ow = 0; itr_ow < out_width; itr_ow++)
  {
    CIRC_BUF_ADD_COLS
      (cols_added
      ,cols_to_add
      ,left_pad
      ,right_pad
      ,input_col
      ,input_height
      ,input_width
      ,input_channels
      ,kernel_height
      ,kernel_width
      ,channels_multiplier
      ,x_stride
      ,x_padding
      ,y_padding
      ,out_height
      ,p_circ_buf
      ,pt_inp
      );

    p_inp_circ = (WORD8 *)p_circ_buf->p_curr;

    conv2d_nhwc_8x8
      ((pWORD8)(&p_out[itr_ow*input_channels*channels_multiplier])
      ,p_kernel
      ,p_inp_circ
      ,p_bias
      ,kernel_height
      ,kernel_width
      ,out_height
      ,out_width
      ,(input_channels * channels_multiplier)
      ,x_stride
      ,y_stride
      ,acc_shift
      ,bias_shift
      ,p_scratch
      );
  }
}

WORD32 xa_nn_conv2d_depthwise_8x8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD8 *__restrict__ p_inp
  ,const WORD8 *__restrict__ p_bias
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
  ,WORD32  acc_shift
  ,WORD32  bias_shift
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  if(inp_data_format == 0)
  {
    xa_nn_conv2d_depthwise_nhwc_8x8
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
      ,acc_shift
      ,bias_shift
      ,p_scratch
      );
  }
  else if(inp_data_format == 1)
  {
    xa_nn_conv2d_depthwise_nchw_8x8
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
      ,acc_shift
      ,bias_shift
      ,p_scratch
      );
  }
  return 0;
}
