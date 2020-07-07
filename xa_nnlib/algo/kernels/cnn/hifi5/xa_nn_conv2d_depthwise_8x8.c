/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
#include "common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

/* 2D Convolution implementation */
static inline void conv2d_8x8_hf5_convmul
  (pWORD8 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
  ,pWORD8 __restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
  ,pWORD8 __restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
  ,WORD8 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
  ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
  ,int out_stride
  ,int x_stride
  ,int y_stride
  ,int acc_shift
  ,int bias_shift
  ,pWORD64 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
  )
{
  /* Importance of actual_out_width, since we are appending zeros input left
   * and right side. No problem with left padding, but for right padding that
   * is done to make sure that input_width is multiple of 4. Here
   * 'output_width_for_x_stride_1' value is calculated based on this padded value. But
   * actually expected output width to pick correct values from 'output_width_for_x_stride_1' on
   * jumps of 'x_stride'. */

  int kernel_width_pad = (kernel_width+3)&(~3);

  /* Generic case */
  int i, j, k, l;
  int output_height = input_height - kernel_height + 1;
  int output_width_for_x_stride_1;

  /* Here input_width is nothing but circ_buf_width, which is taken care to be
   * multiple of 4. */
  output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
  /* output_width_for_x_stride_1 loop is unrolled by 4 so keeping this dimension to multiple of 4 */
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));

  /* Please note that below addition of 1 is done to adjust in C style indices
   * */
  if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
  {
    return;
  }
  if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
  {
    return;
  }

  ae_int64 accu_int64_0, accu_int64_1, accu_int64_2, accu_int64_3;
  ae_int64 accu_int64_4, accu_int64_5, accu_int64_6, accu_int64_7;
  ae_int32x2 accu_int32x2_0, accu_int32x2_1, accu_int32x2_2, accu_int32x2_3;
  ae_int64 *scratch_ptr = (ae_int64 *)p_scratch;

  /* Below handling is done to load 8b bias into Xtensa datatype. */
  ae_int16 _ae_int16_bias;
  ae_int16 *_ae_int16_p_bias = &_ae_int16_bias;
  (*((WORD16 *) _ae_int16_p_bias)) = bias;

  ae_int64 _ae_int64_sat_bias;
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) (_ae_int16_bias)), bias_shift);

  if(kernel_width_pad==12)
  {
      ae_int8x8 d_inp00, d_inp01, d_inp02;
      ae_int8x8 d_inp10, d_inp11, d_inp12;
      ae_int8x8 d_ker0, d_ker1, d_ker2;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp=output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          accu_int32x2_2 = AE_ZERO32();
          accu_int32x2_3 = AE_ZERO32();
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
          ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + 1) * input_width) + j));
          ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
          ae_valign ker_a = AE_LA64_PP(pt_ker);
          for(k = 0; k < (kernel_height>>1); k++)
          {
            AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
            AE_LA8X8_IP(d_ker2, ker_a, pt_ker);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, 8);
            AE_L8X8_XC(d_inp02, pt_inp0, (2*input_width-16));
            AE_L8X8_XC(d_inp10, pt_inp1, 8);
            AE_L8X8_XC(d_inp11, pt_inp1, 8);
            AE_L8X8_XC(d_inp12, pt_inp1, (2*input_width-16));

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            AE_MULA8Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp00, d_inp01);
            AE_MULA2X4Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp01, d_inp10);
            AE_MULA2X4Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp01, d_inp02, d_inp10, d_inp11);
            AE_MULA8Q8X8CNV_L(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp10, d_inp11);
            AE_MULA8Q8X8CNV_H(accu_int32x2_2, accu_int32x2_3, d_ker2, d_inp11, d_inp12);
          }
          if(kernel_height&1)
          {
            AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
            /* 4 values outside kernel are loaded but it is safe because second
            row of input is 0, i.e. id10, id11 are initialized to 0 */
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, 8);
            AE_L8X8_XC(d_inp02, pt_inp0, 8);
            d_inp10 = d_inp11 = AE_MOVINT8X8_FROMINT64(AE_ZERO64());

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            AE_MULA8Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp00, d_inp01);
            AE_MULA2X4Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp01, d_inp10);
            AE_MULA2X4Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp01, d_inp02, d_inp10, d_inp11);
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          AE_ADDW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
          AE_ADDW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        for(j = temp; j < output_width_for_x_stride_1; j+=4)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
          ae_int8x8 *pt_inp1 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, sizeof(WORD8) * (((i * y_stride + 1) * input_width) + j));
          ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
          ae_valign ker_a = AE_LA64_PP(pt_ker);
          for(k = 0; k < (kernel_height>>1); k++)
          {
            AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
            AE_LA8X8_IP(d_ker2, ker_a, pt_ker);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, (2*input_width-8));
            AE_L8X8_XC(d_inp10, pt_inp1, 8);
            AE_L8X8_XC(d_inp11, pt_inp1, (2*input_width-8));

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            AE_MULA2X4Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp01, d_inp10);
            AE_MULA8Q8X8CNV_L(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp10, d_inp11);
          }
          if(kernel_height&1)
          {
            AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
            /* 4 values outside kernel are loaded but it is safe because second
            row of input is 0, i.e. id10, id11 are initialized to 0 */
            AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, 8);
            d_inp10 = AE_MOVINT8X8_FROMINT64(AE_ZERO64());

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            AE_MULA2X4Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp01, d_inp10);
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else if(kernel_width_pad==8)
  {
      /* Regression is passing, but runperf.sh case is not matching with ref output,
      it is most probably due to kernel not being properly padded, need to fix this
      in testbench */
      ae_int8x8 d_inp00, d_inp01;
      ae_int8x8 d_ker0;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp=output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          accu_int32x2_2 = AE_ZERO32();
          accu_int32x2_3 = AE_ZERO32();
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
          ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
          for(k = 0; k < kernel_height; k++)
          {
            AE_L8X8_IP(d_ker0, pt_ker, 8);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, (input_width-8));

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            AE_MULA8Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp00, d_inp01);
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          AE_ADDW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
          AE_ADDW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        for(j = temp; j < output_width_for_x_stride_1; j+=4)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
          ae_int8x8 *pt_ker = (ae_int8x8 *)p_ker;
#pragma loop_count min=1
          for(k = 0; k < kernel_height; k++)
          {
            AE_L8X8_IP(d_ker0, pt_ker, 8);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
            AE_L8X8_XC(d_inp01, pt_inp0, (input_width-8));

            AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else
  {
      ae_int8x8 d_inp00, d_inp01;
      ae_int8x8 d_ker0;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp=output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          accu_int32x2_2 = AE_ZERO32();
          accu_int32x2_3 = AE_ZERO32();
          for(k = 0; k < kernel_height; k++)
          {
            ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j));
            ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[k*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker0);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
#pragma no_unroll
            for(l = 0; l < (kernel_width_pad>>3); l++)
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker0);
              AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
              AE_MULA8Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp00, d_inp01);
              d_inp00 = d_inp01;
            }
            if(kernel_width_pad&4)
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker0);
              /* Last 4 value are not from kernel so making them 0 */
              d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));
              AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
              AE_MULA8Q8X8CNV_L(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp00, d_inp01);
            }
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          AE_ADDW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
          AE_ADDW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int32x2_0 = AE_ZERO32();
          accu_int32x2_1 = AE_ZERO32();
          for(k = 0; k < kernel_height; k++)
          {
            ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * (((i * y_stride + k) * input_width) + j));
            ae_int8x8 *pt_ker0 = (ae_int8x8 *)(&p_ker[k*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker0);
            AE_L8X8_XC(d_inp00, pt_inp0, 8);
#pragma no_unroll
            for(l = 0; l < (kernel_width_pad>>3); l++)
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker0);
              AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
              d_inp00 = d_inp01;
            }
            if(kernel_width_pad&4)
            {
              AE_L8X8_XC(d_inp01, pt_inp0, 8);
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker0);
              /* Last 4 value are not from kernel so making them 0 */
              d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));
              AE_MULA8Q8X8CNV_H(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01);
            }
          }
          AE_ADDW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
          AE_ADDW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }

  /* Here we store output based on strides. For values in a row, values
   * will be picked from it as per 'x_stride'. No need to worry about
   * height dimension, since we took care of it by efficient row
   * accesses. */
  scratch_ptr = (ae_int64 *) p_scratch;

  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
    WORD8 *out_ptr  = (WORD8 *) p_out + (i * out_stride * actual_out_width);
    ae_int32x2 accu_int32x2_0;
    for(j = 0; j < actual_out_width; j++)
    {
      accu_int64_0 = scratch_ptr[(j * x_stride)];
      accu_int64_0 = AE_ADD64S(accu_int64_0, _ae_int64_sat_bias);
      accu_int64_0 = AE_SLAA64S(accu_int64_0, acc_shift);
      accu_int32x2_0 = AE_ROUND32F64SSYM(accu_int64_0);
      ae_int8x8 temp_int8x8_0 = AE_SAT8X4X32_L(accu_int32x2_0, accu_int32x2_0);
      *(ae_int8 *)(&out_ptr[(j * out_stride)]) = AE_MOVINT8_FROMINT8X8(temp_int8x8_0);
    }
  }
}

WORD32 xa_nn_conv2d_depthwise_8x8
    (pWORD8 __restrict__ p_out
     ,pWORD8 __restrict__ p_kernel
     ,pWORD8 __restrict__ p_inp
     ,pWORD8 __restrict__ p_bias
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND((y_stride > kernel_height), -1);
  XA_NNLIB_ARG_CHK_COND((x_stride > kernel_width), -1);

  xa_nn_conv2d_depthwise_init
    (p_scratch
     ,input_width
     ,kernel_height
     ,kernel_width
     ,x_stride
     ,y_stride
     ,x_padding
     ,out_width
     ,8
    );

  xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
  int itr_ic, itr_cm, itr_oh;
  int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
  int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
  int rows_to_add, top_pad, bottom_pad, rows_added;
  int input_row;
  pWORD8 pt_ker;
  pWORD8 pt_inp;
  pWORD8 p_inp_circ;
  p_scratch = (pWORD64)(p_state->p_scratch);

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  WORD8 bias = 0;

  //ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD8, WORD16);
  acc_shift =acc_shift + 32;

  for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
  {
    pt_inp = &p_inp[itr_ic*input_height*input_width];
    for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
    {
      pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width_pad];
      bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

      CIRC_BUF_ADD_ROWS_INIT(rows_added
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

      for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
      {
        CIRC_BUF_ADD_ROWS(rows_added
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

        conv2d_8x8_hf5_convmul
          ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
           ,pt_ker
           ,p_inp_circ
           ,bias
           ,p_circ_buf->rows
           ,p_circ_buf->row_offset
           ,kernel_height
           ,kernel_width
           ,circ_out_height
           ,out_width
           ,(input_channels * channels_multiplier)
           ,x_stride
           ,y_stride
           ,acc_shift
           ,bias_shift
           ,p_scratch
          );
      }

      CIRC_BUF_ADD_ROWS(rows_added
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

      conv2d_8x8_hf5_convmul
        ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
         ,pt_ker
         ,p_inp_circ
         ,bias
         ,p_circ_buf->rows
         ,p_circ_buf->row_offset
         ,kernel_height
         ,kernel_width
         ,(out_height - itr_oh)
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

  return 0;
}
