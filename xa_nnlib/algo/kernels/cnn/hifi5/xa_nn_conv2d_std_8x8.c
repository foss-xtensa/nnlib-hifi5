/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_err_chk.h"


static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    WORD8 *p_bias,
    WORD8 *p_out,
    WORD32 bias_shift,
    WORD32 acc_shift)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i=0;i<out_height;i++)
  {
    for(j=0;j<out_width_over_x_pad;j++)
    {
      for(k=0;k<out_channels;k++)
      {

        ae_int64 acc = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[k]));
        acc = AE_SLAA64S(acc, 8);
        acc = AE_SLAA64S(acc, -56);
        acc = AE_SLAA64S(acc, bias_shift);
        acc = AE_SLAA64S(acc, acc_shift);
        ae_int8x8 res2 = AE_MOVINT8X8_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(acc),24),-24));
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] =(WORD8)AE_MOVAD8(res2,0);
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    WORD8 *p_bias,
    WORD8 *p_out,
    WORD32 bias_shift,
    WORD32 acc_shift)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i=0;i<out_height;i++)
  {
    for(j=idx_out_width_over_x_r_pad;j<out_width;j++)
    {
      for(k=0;k<out_channels;k++)
      {
        ae_int64 acc = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[k]));
        acc = AE_SLAA64S(acc, 8);
        acc = AE_SLAA64S(acc, -56);
        acc = AE_SLAA64S(acc, bias_shift);
        acc = AE_SLAA64S(acc, acc_shift);
        ae_int8x8 res2 = AE_MOVINT8X8_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(acc),24),-24));
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] = (WORD8)AE_MOVAD8(res2,0);
      }
    }
  }
  return out_width_over_x_r_pad;
}

WORD32 xa_nn_conv2d_std_8x8(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_inp,
    WORD8* __restrict__ p_kernel,
    WORD8* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 bias_shift,
    WORD32 acc_shift,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  /* Implementation dependent checks */
  XA_NNLIB_ARG_CHK_COND((x_stride > kernel_width), -1);

  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv2d_std_init_state((void*)p_state,(void*)p_kernel,input_height,input_channels,kernel_height,kernel_width,x_stride,y_stride,y_padding,out_height,input_bytewidth*8);

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;

  // Limit effective bias_shift and acc_shift to [-63 ... 63]
  // +16 to conform with 8bit left shifts of 8bit kernel and input loads
  //bias_shift = bias_shift + 16;
  bias_shift = bias_shift > 63 ? 63 : bias_shift < -63 ? -63 : bias_shift;
  /* +48 to move acc to upper 16bits, as TRUNC keeps upper 32bits and ROUND keeps upper 16bits;
     -16 to remove 8bit left shifts of kernel and input */
  acc_shift = acc_shift + 32;
  acc_shift = acc_shift > 63 ? 63 : acc_shift < -63 ? -63 : acc_shift;


  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= kernel_width)
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, bias_shift, acc_shift);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }


  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, bias_shift, acc_shift);
  }


  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf(input_channels, input_channels, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;


  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf(input_channels, input_channels, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_stride;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_8x8_8_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_kernel /* vec: cols */
       ,p_bias /* bias */
       ,out_height /* rows */
       ,input_channels * kernel_width * kernel_height /* cols */
       ,input_channels * kernel_width * y_stride/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels * kernel_width * kernel_height /* vec_offset */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,bias_shift
       ,acc_shift
      );

    p_out += out_width_offset;
  }

  return 0;
}

