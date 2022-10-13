/*******************************************************************************
* Copyright (c) 2022 Cadence Design Systems, Inc.
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
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

#define INTERLEAVE_6(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, src0, src1, src2, src3, src4, src5) \
 { \
   ae_int8x8 tmp01_0, tmp01_1, tmp23_0, tmp23_1, tmp0123_0, tmp0123_1, tmp0123_2, tmp0123_3, tmp45_0, tmp45_1; \
   AE_DSEL8X8(tmp01_0, tmp01_1, src0, src1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xf7e6d5c4, 0xb3a29180)));\
   AE_DSEL8X8(tmp23_0, tmp23_1, src2, src3, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xf7e6d5c4, 0xb3a29180)));\
   AE_DSEL8X8(tmp45_0, tmp45_1, src4, src5, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xf7e6d5c4, 0xb3a29180)));\
   AE_DSEL8X8(tmp0123_0, tmp0123_1, tmp01_0, tmp23_0, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfe76dc54, 0xba329810)));\
   AE_DSEL8X8(tmp0123_2, tmp0123_3, tmp01_1, tmp23_1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfe76dc54, 0xba329810)));\
   AE_DSEL8X8(dst0, dst1, tmp0123_0, tmp45_0, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc7600, 0xba984500)));\
   AE_DSEL8X8(dst2, dst3, tmp0123_1, tmp45_0, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc3200, 0xba981000)));\
   AE_DSEL8X8(dst4, dst5, tmp0123_2, tmp45_1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc7600, 0xba984500)));\
   AE_DSEL8X8(dst6, dst7, tmp0123_3, tmp45_1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc3200, 0xba981000)));\
 }

#define INTERLEAVE_5(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, src0, src1, src2, src3, src4) \
 { \
   ae_int8x8 tmp01_0, tmp01_1, tmp23_0, tmp23_1, tmp0123_0, tmp0123_1, tmp0123_2, tmp0123_3, tmp45_0, tmp45_1; \
   AE_DSEL8X8(tmp01_0, tmp01_1, src0, src1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xf7e6d5c4, 0xb3a29180)));\
   AE_DSEL8X8(tmp23_0, tmp23_1, src2, src3, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xf7e6d5c4, 0xb3a29180)));\
   AE_DSEL8X8(tmp0123_0, tmp0123_1, tmp01_0, tmp23_0, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfe76dc54, 0xba329810)));\
   AE_DSEL8X8(tmp0123_2, tmp0123_3, tmp01_1, tmp23_1, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfe76dc54, 0xba329810)));\
   AE_DSEL8X8(dst0, dst1, tmp0123_0, src4, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc7000, 0xba986000)));\
   AE_DSEL8X8(dst2, dst3, tmp0123_1, src4, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc5000, 0xba984000)));\
   AE_DSEL8X8(dst4, dst5, tmp0123_2, src4, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc3000, 0xba982000)));\
   AE_DSEL8X8(dst6, dst7, tmp0123_3, src4, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfedc1000, 0xba980000)));\
 }

#define INTERLEAVE_3(dst0, dst1, src0, src1, src2) \
 { \
   ae_int8x8 tmp01_0; \
   tmp01_0 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src0), AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e060c04, 0x0a020800)));\
   AE_DSEL8X8(dst0, dst1, tmp01_0, AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfe60dc40, 0xba209800)));\
 }

#define PACK_32X2(dst1, src1, src2) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

#define SET_INP_PTR_X_Y(ptr, x, y) \
          if((x < x_padding) || (x >= x_padding + input_width)) \
            ptr = p_dummy_inp; \
          else \
            ptr = (ae_int8x8 *)(p_inp + (y - y_padding) * input_width * input_channels + (x - x_padding) * input_channels); \

#define SET_INP_PTR_Y(ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, y) \
        if((y < y_padding) || (y >= y_padding + input_height)) \
        { \
          ptr0 = ptr1 = ptr2 = ptr3 = ptr4 = ptr5 = p_dummy_inp; \
        } \
        else \
        { \
          int x_idx = itr_ow * x_stride; \
          SET_INP_PTR_X_Y(ptr0, x_idx, y); x_idx++; \
          SET_INP_PTR_X_Y(ptr1, x_idx, y); x_idx++; \
          SET_INP_PTR_X_Y(ptr2, x_idx, y); x_idx++; \
          SET_INP_PTR_X_Y(ptr3, x_idx, y); x_idx++; \
          SET_INP_PTR_X_Y(ptr4, x_idx, y); x_idx++; \
          SET_INP_PTR_X_Y(ptr5, x_idx, y); x_idx++; \
        } \

/* 2D Convolution implementation */
static inline void conv2d_nchw_asym8xasym8_hf5_convmul
  (pUWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_ker
  ,const UWORD8 *__restrict__ p_inp
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
  ,WORD32  kernel_zero_bias
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
  ae_int32x2 inp_ker_zb = AE_MOVDA32X2(-input_zero_bias, -kernel_zero_bias);
  AE_MOVZBVCDR(AE_MOVINT64_FROMINT32X2(inp_ker_zb));

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
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
        ae_int8x8 *pt_inp0 = (ae_int8x8 *)(p_inp);
        AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, sizeof(WORD8) * ((i * y_stride * input_width) + j));
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
          AE_L8X8_XC(d_inp02, pt_inp0, (input_width - 16));
          AE_L8X8_XC(d_inp10, pt_inp0, 8);
          AE_L8X8_XC(d_inp11, pt_inp0, 8);
          AE_L8X8_XC(d_inp12, pt_inp0, (input_width - 16));

          AE_MULAUUZB8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          AE_MULAUUZB8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          AE_MULAUUZB2X4Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker1, d_inp01, d_inp10);
          AE_MULAUUZB2X4Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker1, d_inp01, d_inp02, d_inp10, d_inp11);
          AE_MULAUUZB8Q8X8CNV_L(d_acc32_0, d_acc32_1, d_ker2, d_inp10, d_inp11);
          AE_MULAUUZB8Q8X8CNV_H(d_acc32_2, d_acc32_3, d_ker2, d_inp11, d_inp12);
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
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
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

          AE_MULAUUZB8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
          AE_MULAUUZB8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
          AE_MULAUUZB8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
          AE_MULAUUZB8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
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
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
        d_acc32_4 = AE_ZERO32();
        d_acc32_5 = AE_ZERO32();
        d_acc32_6 = AE_ZERO32();
        d_acc32_7 = AE_ZERO32();

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

          AE_MULAUUZB2X4Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp10);
          AE_MULAUUZB2X4Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01, d_inp10, d_inp11);
          AE_MULAUUZB2X4Q8X8CNV_H(d_acc32_4, d_acc32_5, d_ker0, d_inp01, d_inp11);
          AE_MULAUUZB2X4Q8X8CNV_L(d_acc32_6, d_acc32_7, d_ker0, d_inp01, d_inp02, d_inp11, d_inp12);
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
        d_acc32_0 = AE_ZERO32();
        d_acc32_1 = AE_ZERO32();
        d_acc32_2 = AE_ZERO32();
        d_acc32_3 = AE_ZERO32();
#pragma loop_count min=1
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
            AE_MULAUUZB8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker0, d_inp00, d_inp01);
            AE_MULAUUZB8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
            AE_MULAUUZB8Q8X8CNV_H(d_acc32_0, d_acc32_1, d_ker1, d_inp10, d_inp11);
            AE_MULAUUZB8Q8X8CNV_L(d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
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
              AE_MULAUUZB4O8X8CNV_H(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker0, d_inp00, d_inp01);
              AE_MULAUUZB4O8X8CNV_L(d_acc32_0, d_acc32_1, d_acc32_2, d_acc32_3, d_ker1, d_inp10, d_inp11);
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
      AE_MINMAX32(accu_int32_0, AE_ZERO32(), AE_MOVDA32(255));
      accu_int8x8 = AE_SATU8X4X32_L(accu_int32_0, accu_int32_0);

      *(ae_int8 *)(&out_ptr[((j + 1) * out_stride)]) = AE_MOVINT8_FROMINT8X8(accu_int8x8);
      *(ae_int8 *)(&out_ptr[(j * out_stride)]) = AE_MOVINT8_FROMINT8X8(AE_SEL8X8I(accu_int8x8, accu_int8x8, 19));
    }
    if(j < actual_out_width)
    {
      accu_int32_0 = scratch_ptr1[(j * x_stride)];

      accu_int32_0 = AE_ADD32S(accu_int32_0, d_bias);
      MPY_BY_QUANT_MULT_X2_OUT32(accu_int32_0, accu_int32_0, out_multiplier, left_shift, right_shift);
      accu_int32_0 = AE_ADD32S(accu_int32_0, AE_MOVDA32X2(out_zero_bias, out_zero_bias));
      AE_MINMAX32(accu_int32_0, AE_ZERO32(), AE_MOVDA32(255));
      accu_int8x8 = AE_SATU8X4X32_L(accu_int32_0, accu_int32_0);

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
      ae_int64 sel = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0x0f0e0d0c, 0x0b0a0908)); \
      ae_int8x8 sel8 = AE_MOVINT8X8_FROMINT64(sel); \
      sel8 = AE_ADD8(sel8, AE_MOVDA8(8 - (kw & 7))); \
      sel = AE_MOVINT64_FROMINT8X8(sel8); \
      sel = AE_SLAA64(sel, 8 * (8 - (kw & 7))); \
      sel8 = AE_MOVINT8X8_FROMINT64(sel); \
      d_tmp = AE_SEL8X8(d_tmp, AE_MOVDA8(-kernel_zero_bias), sel8); \
      AE_SA8X8_IP(d_tmp, out_a, pae_out); \
    } \
    AE_SA64POS_FP(out_a, pae_out); \
  } \
}

static void xa_nn_conv2d_depthwise_nchw_asym8xasym8
  (pUWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_kernel
  ,const UWORD8 *__restrict__ p_inp
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
  ,WORD32  kernel_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
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
  const UWORD8 *pt_ker;
  const WORD8 *pt_inp;
  UWORD8 *p_inp_circ;
  int i;
  UWORD8 *p_kernel_padded = (UWORD8 *)(p_state->p_scratch);
  p_kernel_padded = (UWORD8 *)ALIGN_PTR(p_kernel_padded, 8);
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
    pae_ker_pad[i] = AE_MOVDA8(-kernel_zero_bias);
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

      p_inp_circ = (UWORD8 *)p_circ_buf->p_curr;

#pragma loop_count min=1
      for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
      {
        pt_ker = (const UWORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
        COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
        bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

        conv2d_nchw_asym8xasym8_hf5_convmul
          ((UWORD8 *)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
          ,kernel_zero_bias
          ,out_multiplier
          ,out_shift
          ,out_zero_bias
          ,p_tmp_out
          );
      }
    }
  }
}

#ifndef AE_MULUUZB3X3O8X8
/* 2D Convolution implementation */
static inline void conv2d_nhwc_asym8xasym8
  (pWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_ker
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
  ,WORD32  kernel_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
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

#if TFLITE_SINGLE_ROUNDING
  int left_shift = out_shift;
  int right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift = XT_MAX(0, out_shift);
  int right_shift = XT_MAX(0, -out_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */

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
    int ysXkwXic = y_stride * kernel_width * inp_channels_pad;
#pragma loop_count min=1
    for(itr_ch = 0; itr_ch < out_channels; itr_ch += 8)
    {
      pt_inp0 = (ae_int8x8 *)p_inp;
      AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, itr_ch + itr_oh * ysXkwXic);
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
        AE_L8X8_XC(d_inp0, pt_inp0, ysXkwXic);
        AE_L8X8_XC(d_inp1, pt_inp0, inp_channels_pad - ysXkwXic);
        AE_LA8X8_IP(d_ker, ker_a, ptae_ker);
        ptae_ker = (ae_int8x8 *)((WORD8 *)ptae_ker + (ker_channels_pad - 8));
        ker_a = AE_LA64_PP(ptae_ker);
        AE_SUBW8U(d_inp00, d_inp01, d_inp0, AE_MOVDA8(-input_zero_bias));
        AE_SUBW8U(d_inp10, d_inp11, d_inp1, AE_MOVDA8(-input_zero_bias));
        AE_SUBW8U(d_ker0, d_ker1, d_ker, AE_MOVDA8(-kernel_zero_bias));
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
      ae_int16x4 d_acc16_0, d_acc16_1;
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_acc16_0, d_acc0, d_acc1, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_acc16_1, d_acc2, d_acc3, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(d_acc16_0, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(d_acc16_1, AE_ZERO16(), AE_MOVDA16(255));

      d_acc8x8 = AE_SATU8X8X16(d_acc16_0, d_acc16_1);
      if(out_channels - itr_ch >= 8)
      {
#pragma frequency_hint FREQUENT
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

      d_acc4 = AE_ADD32S(d_acc4, d_bias0);
      d_acc5 = AE_ADD32S(d_acc5, d_bias1);
      d_acc6 = AE_ADD32S(d_acc6, d_bias2);
      d_acc7 = AE_ADD32S(d_acc7, d_bias3);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_acc16_0, d_acc4, d_acc5, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d_acc16_1, d_acc6, d_acc7, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(d_acc16_0, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(d_acc16_1, AE_ZERO16(), AE_MOVDA16(255));

      d_acc8x8 = AE_SATU8X8X16(d_acc16_0, d_acc16_1);
      if(out_height-itr_oh >= 2)
      {
#pragma frequency_hint FREQUENT
        if(out_channels-itr_ch >= 8)
        {
#pragma frequency_hint FREQUENT
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
static inline void conv2d_nhwc_asym8xasym8
  (pWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_ker
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
  ,WORD32  kernel_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
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
  ae_valignx2 bias_a;
  ae_int32x2 d_acc0, d_acc1, d_acc2, d_acc3;
  ae_int32x2 d_acc4, d_acc5, d_acc6, d_acc7;
  ae_int32x2 d_acc8, d_acc9, d_acc10, d_acc11;
  ae_int32x2 d_acc12, d_acc13, d_acc14, d_acc15;
  
  ae_int32x2 d_bias0, d_bias1, d_bias2, d_bias3;
  ae_int32x2 d_bias4, d_bias5, d_bias6, d_bias7;

#if TFLITE_SINGLE_ROUNDING
  int left_shift = out_shift;
  int right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  int left_shift = XT_MAX(0, out_shift);
  int right_shift = XT_MAX(0, -out_shift);
#endif /* #if TFLITE_SINGLE_ROUNDING */
  
  inp_channels_pad = (out_channels + 15)&(~15);
  ker_channels_pad = (out_channels + 15)&(~15);

  /*Load bias value in state registers*/
  ae_int64 biasvc1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-input_zero_bias, -kernel_zero_bias));
  AE_MOVZBVCDR(biasvc1);

  ae_int8x8 neg_kerzb = AE_MOVDA8(-kernel_zero_bias);

/*Special case 3x3 code with HiFi5 RI6 ISA*/
#if 1  
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
        AE_MULUUZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, d_ker10, d_ker20, d_inp000, d_inp010, d_inp020);

        /* Apply second row of kernel */
        AE_L8X8_XC(d_inp000, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp010, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp020, pt_inp0, inp_channels_pad);
        AE_MULAUUZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker30, d_ker40, d_ker50, d_inp000, d_inp010, d_inp020);

        /* Apply third row of kernel */
        AE_L8X8_XC(d_inp000, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp010, pt_inp0, inp_channels_pad);
        AE_L8X8_XC(d_inp020, pt_inp0, ysXkwXic - 8*inp_channels_pad);
        AE_MULAUUZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker60, d_ker70, d_ker80, d_inp000, d_inp010, d_inp020);
        
        d_acc0  = AE_ADD32S(d_acc0, d_bias0);
        d_acc1  = AE_ADD32S(d_acc1, d_bias1);
        d_acc2  = AE_ADD32S(d_acc2, d_bias2);
        d_acc3  = AE_ADD32S(d_acc3, d_bias3);

        ae_int16x4 out0, out1;
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out0, d_acc0, d_acc1, out_multiplier, left_shift, right_shift, out_zero_bias);
        MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out1, d_acc2, d_acc3, out_multiplier, left_shift, right_shift, out_zero_bias);

        AE_MINMAX16(out0, AE_ZERO16(), AE_MOVDA16(255));
        AE_MINMAX16(out1, AE_ZERO16(), AE_MOVDA16(255));

        ae_int8x8 d_acc8x8_0 = AE_SATU8X8X16(out0, out1);

        AE_SAV8X8X2_XP(d_acc8x8_0, d_acc8x8_0, out0_a, ae_out_ptr0, XT_MIN(out_channels - itr_ch, 8));
        AE_SA128POS_FP(out0_a, ae_out_ptr0);
      }
    }
  }
  else
  {
#endif
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

        AE_MULAUUZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, d_ker10, d_ker20, d_inp000, d_inp010, d_inp020);
        AE_MULAUUZB3X3O8X8(d_acc4, d_acc5, d_acc6, d_acc7, d_ker01, d_ker11, d_ker21, d_inp001, d_inp011, d_inp021);
        AE_MULAUUZB3X3O8X8(d_acc8, d_acc9, d_acc10, d_acc11, d_ker00, d_ker10, d_ker20, d_inp100, d_inp110, d_inp120);
        AE_MULAUUZB3X3O8X8(d_acc12, d_acc13, d_acc14, d_acc15, d_ker01, d_ker11, d_ker21, d_inp101, d_inp111, d_inp121);
      }
      for(itr_kw = loop_count * 3; itr_kw < kernel_height * kernel_width; itr_kw++)
      {
        AE_L8X8X2_XC(d_inp000, d_inp001, pt_inp0, inp_channels_pad);
        AE_L8X8X2_XC(d_inp100, d_inp101, pt_inp1, inp_channels_pad);
        AE_L8X8X2_XP(d_ker00, d_ker01, ptae_ker, ker_channels_pad);
        AE_MULAUUZB3X3O8X8(d_acc0, d_acc1, d_acc2, d_acc3, d_ker00, neg_kerzb, neg_kerzb, d_inp000, d_inp000, d_inp000);
        AE_MULAUUZB3X3O8X8(d_acc4, d_acc5, d_acc6, d_acc7, d_ker01, neg_kerzb, neg_kerzb, d_inp001, d_inp001, d_inp001);
        AE_MULAUUZB3X3O8X8(d_acc8, d_acc9, d_acc10, d_acc11, d_ker00, neg_kerzb, neg_kerzb, d_inp100, d_inp100, d_inp100);
        AE_MULAUUZB3X3O8X8(d_acc12, d_acc13, d_acc14, d_acc15, d_ker01, neg_kerzb, neg_kerzb, d_inp101, d_inp101, d_inp101);
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
      
      ae_int16x4 out0, out1, out2, out3;
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out0, d_acc0, d_acc1, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out1, d_acc2, d_acc3, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out2, d_acc4, d_acc5, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out3, d_acc6, d_acc7, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(out0, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out1, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out2, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out3, AE_ZERO16(), AE_MOVDA16(255));

      ae_int8x8 d_acc8x8_0 = AE_SATU8X8X16(out0, out1);
      ae_int8x8 d_acc8x8_1 = AE_SATU8X8X16(out2, out3);
      if(out_channels - itr_ch >= 16)
      {
          AE_SA8X8X2_IP(d_acc8x8_0, d_acc8x8_1, out0_a, ae_out_ptr0);
      }
      else
      {        
        AE_SAV8X8X2_XP(d_acc8x8_0, d_acc8x8_1, out0_a, ae_out_ptr0, out_channels - itr_ch);
      }
      AE_SA128POS_FP(out0_a, ae_out_ptr0);

      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out0, d_acc8, d_acc9, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out1, d_acc10, d_acc11, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out2, d_acc12, d_acc13, out_multiplier, left_shift, right_shift, out_zero_bias);
      MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out3, d_acc14, d_acc15, out_multiplier, left_shift, right_shift, out_zero_bias);

      AE_MINMAX16(out0, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out1, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out2, AE_ZERO16(), AE_MOVDA16(255));
      AE_MINMAX16(out3, AE_ZERO16(), AE_MOVDA16(255));

      d_acc8x8_0 = AE_SATU8X8X16(out0, out1);
      d_acc8x8_1 = AE_SATU8X8X16(out2, out3);
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
#if 1 /*Special case 3x3 code with HiFi5 RI6 ISA*/
  }
#endif
  }
}
#endif

static void xa_nn_conv2d_depthwise_nhwc_asym8xasym8
  (pUWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_kernel
  ,const UWORD8 *__restrict__ p_inp
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
  ,WORD32  kernel_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
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
  const UWORD8 *pt_ker = p_kernel;

  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);

  pt_inp = (const WORD8 *)p_inp;

#ifdef AE_MULUUZB3X3O8X8
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
  pt_ker = (const UWORD8 *)p_circ_buf->p_end;
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

    conv2d_nhwc_asym8xasym8
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
      ,kernel_zero_bias
      ,out_multiplier
      ,out_shift
      ,out_zero_bias
      ,p_scratch
      );
  }
}

WORD32 xa_nn_conv2d_depthwise_asym8xasym8
  (pUWORD8 __restrict__ p_out
  ,const UWORD8 *__restrict__ p_kernel
  ,const UWORD8 *__restrict__ p_inp
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
  ,WORD32  kernel_zero_bias
  ,WORD32  out_multiplier
  ,WORD32  out_shift
  ,WORD32  out_zero_bias
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
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias > 0 || input_zero_bias < -255), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_zero_bias > 0 || kernel_zero_bias < -255), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift>31), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  if(inp_data_format == 0)
  {
    xa_nn_conv2d_depthwise_nhwc_asym8xasym8
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
      ,kernel_zero_bias
      ,out_multiplier
      ,out_shift
      ,out_zero_bias
      ,out_data_format
      ,p_scratch
      );
  }
  else if(inp_data_format == 1)
  {
    xa_nn_conv2d_depthwise_nchw_asym8xasym8
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
      ,kernel_zero_bias
      ,out_multiplier
      ,out_shift
      ,out_zero_bias
      ,out_data_format
      ,p_scratch
      );
  }
  return 0;
}
