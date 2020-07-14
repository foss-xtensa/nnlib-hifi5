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
static inline void conv2d_8x16_hf4_convmul
  (pWORD16 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
  ,pWORD8  __restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
  ,pWORD16 __restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
  ,WORD16 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height       /* This is the actual output height, processing should be limited to it. */
  ,int actual_out_width        /* This is the actual output width, processing should be limited to it. */
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
  ae_int16x4 accu_int16x4_0;
  ae_int64 *scratch_ptr = (ae_int64 *)p_scratch;

  ae_int64 _ae_int64_sat_bias;
  _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) (*((ae_int16 *) &bias))), bias_shift);

  if(kernel_width_pad==12)
  {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03, d_inp04;
      ae_int16x4 d_inp10, d_inp11, d_inp12, d_inp13, d_inp14;
      ae_int8x8 d_ker0, d_ker1, d_ker2;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*12
      (which should be <= 256) multiplications */
      kernel_height_acc = 20; // Largest even value less than 256/12 (21.3333)
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
            ae_int16x8 *pt_inp1 = (ae_int16x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, (sizeof(WORD16) * (((i * y_stride + kh + 1) * input_width) + j)));
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < (loop_count>>1); k++)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker2, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, 16);
              AE_L16X4_XC(d_inp04, (ae_int16x4 *)pt_inp0, sizeof(WORD16)*(2*input_width-16));

              AE_L16X4X2_XC(d_inp10, d_inp11, pt_inp1, 16);
              AE_L16X4X2_XC(d_inp12, d_inp13, pt_inp1, 16);
              AE_L16X4_XC(d_inp14, (ae_int16x4 *)pt_inp1, sizeof(WORD16)*(2*input_width-16));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp11, d_inp12, d_inp13);

              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              AE_MULA2X4Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp03, d_inp04, d_inp11, d_inp12);
              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker2, d_inp12, d_inp13, d_inp14);
            }
            if((loop_count)&1)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              /* 4 values outside kernel are loaded but it is safe because second
              row of input is 0, i.e. d_inp10, d_inp11 are initialized to 0 */
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, 16);
              AE_L16X4_XC(d_inp04, (ae_int16x4 *)pt_inp0, sizeof(WORD16)*(2*input_width-16));

              d_inp10 = d_inp11 = d_inp12 = AE_ZERO16();

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);

              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              AE_MULA2X4Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp03, d_inp04, d_inp11, d_inp12);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
            ae_int16x8 *pt_inp1 = (ae_int16x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, (sizeof(WORD16) * (((i * y_stride + kh + 1) * input_width) + j)));
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < (loop_count>>1); k++)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker2, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(2*input_width-8));

              AE_L16X4X2_XC(d_inp10, d_inp11, pt_inp1, 16);
              AE_L16X4X2_XC(d_inp12, d_inp13, pt_inp1, sizeof(WORD16)*(2*input_width-8));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp11, d_inp12, d_inp13);
            }
            if((loop_count)&1)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              /* 4 values outside kernel are loaded but it is safe because second
              row of input is 0, i.e. d_inp10, d_inp11 are initialized to 0 */
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(2*input_width-8));

              d_inp10 = d_inp11 = AE_ZERO16();

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else if(kernel_width_pad==8)
  {
      /* Regression is yet to be tested, but runperf.sh case is not matching with 
      ref output, it is most probably due to kernel not being properly padded,
      need to fix this in testbench */
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int8x8 d_ker0;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*8
      (which should be <= 256) multiplications */
      kernel_height_acc = 32; //256/8
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < loop_count; k++)
            {
              d_ker0 = *pt_ker++;

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(input_width-8));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < loop_count; k++)
            {
              d_ker0 = *pt_ker++;

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4_XC(d_inp02, (ae_int16x4 *)pt_inp0, sizeof(WORD16)*(input_width-8));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else
  {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int8x8 d_ker0;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*kernel_width_pad
      (which should be <= 256) multiplications */
      kernel_height_acc = 256/kernel_width_pad;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
#pragma loop_count min=1
            for(k = 0; k < loop_count; k++)
            {
              ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
              AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh + k) * input_width) + j)));
              ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[(kh+k)*kernel_width_pad]);
              ae_valign ker_a = AE_LA64_PP(pt_ker);
#pragma no_unroll
              for(l = 0; l < (kernel_width_pad>>3); l++)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                AE_L16X4X2_I(d_inp02, d_inp03, pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
                AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              }
              if(kernel_width_pad&4)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
                /* Last 4 value are not from kernel so making them 0 */
                d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                AE_L16X4X2_I(d_inp02, d_inp03, pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
                AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              }
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
#pragma loop_count min=1
            for(k = 0; k < loop_count; k++)
            {
              ae_int16x8 *pt_inp0 = (ae_int16x8 *)(p_inp);
              AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (sizeof(WORD16) * (((i * y_stride + kh + k) * input_width) + j)));
              ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[(kh+k)*kernel_width_pad]);
              ae_valign ker_a = AE_LA64_PP(pt_ker);
#pragma no_unroll
              for(l = 0; l < (kernel_width_pad>>3); l++)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                d_inp02 = AE_L16X4_I((ae_int16x4 *)pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              }
              if(kernel_width_pad&4)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
                /* Last 4 value are not from kernel so making them 0 */
                d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                d_inp02 = AE_L16X4_I((ae_int16x4 *)pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              }
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
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
    ae_int16 *out_ptr  = (ae_int16 *) p_out + (i * out_stride * actual_out_width);

    for(j = 0; j < actual_out_width; j++)
    {
      accu_int64_0 = scratch_ptr[(j * x_stride)];
      accu_int64_0 =  AE_ADD64S(accu_int64_0, _ae_int64_sat_bias);
      accu_int64_0 =  AE_SLAA64S(accu_int64_0, acc_shift);
      accu_int32x2_0 = AE_ROUND32F64SSYM(accu_int64_0);
      accu_int16x4_0 = AE_SAT16X4(accu_int32x2_0, accu_int32x2_0);
      out_ptr[(j * out_stride)] = accu_int16x4_0;
    }
  }
}

WORD32 xa_nn_conv2d_depthwise_8x16
    (pWORD16 __restrict__ p_out
     ,pWORD8 __restrict__ p_kernel
     ,pWORD16 __restrict__ p_inp
     ,pWORD16 __restrict__ p_bias
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
     ,16
    );

  xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
  int itr_ic, itr_cm, itr_oh;
  int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
  int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
  int rows_to_add, top_pad, bottom_pad, rows_added;
  int input_row;
  pWORD8 pt_ker;
  pWORD16 pt_inp;
  pWORD16 p_inp_circ;
  p_scratch = (pWORD64)(p_state->p_scratch);

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  WORD16 bias = 0;

  //ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD16);
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

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

        conv2d_8x16_hf4_convmul
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

      p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

      conv2d_8x16_hf4_convmul
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
