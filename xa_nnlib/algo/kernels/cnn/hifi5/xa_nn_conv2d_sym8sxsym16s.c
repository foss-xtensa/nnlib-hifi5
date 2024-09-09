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
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nn_conv2d_std_state.h"

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
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max)
{
  (VOID) out_zero_bias;
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
        q1 = 0;
        if(pbias != NULL){
          AE_L64_IP(q1, pbias, 8);
        }
        ae_int32x2 acc;
        MPY_BY_QUANT_MULT_ACC64_OUT32(acc, q1, p_out_multiplier[k], p_out_shift[k]);
        d1 = AE_SAT16X4(acc, acc);
        AE_MINMAX16(d1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
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
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_activation_min,
    WORD32 out_activation_max)
{
  (VOID) out_zero_bias;
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;
  ae_int16x4 d1;

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
        q1 = 0;
        if(pbias != NULL){
          AE_L64_IP(q1, pbias, 8);
        }
        ae_int32x2 acc;
        MPY_BY_QUANT_MULT_ACC64_OUT32(acc, q1, p_out_multiplier[k], p_out_shift[k]);
        d1 = AE_SAT16X4(acc, acc);
        AE_MINMAX16(d1, AE_MOVDA16(out_activation_min), AE_MOVDA16(out_activation_max));
        AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
  return out_width_over_x_r_pad;
}

WORD32 xa_nn_conv2d_v2_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 out_activation_min,
    WORD32 out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height!= 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_width!= 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -32768 || out_activation_min > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min || out_activation_max > 32767), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 15), -1);
  }
  const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((out_channels%groups)!=0),-1);
  const int kernels_per_group = out_channels / groups;
  XA_NNLIB_ARG_CHK_COND((kernels_per_group<=0),-1);

  (void)dilation_height;
  (void)dilation_width;
  /* Interchange height and width dimensions when i_h = k_h = o_h = 1 for better throughput */
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;
  if (input_height == 1 && kernel_height == 1 && out_height == 1)
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }
  
  WORD32 j;
  WORD32 input_bytewidth = 2;
  VOID *pp_inp = (VOID *)p_inp;
  WORD16* __restrict__ tmp_out;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;

  WORD32 out_channels_offset = out_data_format ? out_h * out_w : 1;
  WORD32 out_height_offset = out_data_format ? out_w : out_w * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_pad;

  WORD32 input_channels_pad = input_channels;

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias, out_activation_min, out_activation_max);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias, out_activation_min, out_activation_max);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

 for (int grp_i = 0; grp_i < groups; ++grp_i)
  {
    tmp_out=p_out+grp_i*kernels_per_group*out_channels_offset;
    xa_nn_conv2d_std_init_state((void*)p_state
        ,(void*)p_kernel
        ,inp_h
        ,kernel_channels
        ,ker_h
        ,ker_w
        ,x_str,y_str
        ,y_pad
        ,out_h
        ,PREC_SYM16S);
    pp_inp = (VOID *)(p_inp+grp_i*kernel_channels);
    conv2d_group_init_cir_buf(input_channels, input_channels_pad, kernel_channels, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state,0);

    // Index to padded input width
    WORD32 idx_beg_inp_width_pad = ker_w - x_str;
    idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_h x out_channels] per iteration
    for(j=0;j<out_w-out_width_over_x_pad-out_width_over_x_r_pad;j++)
    {
      // Add x_str x (inp_h x input_channels) new planes to circular buffer
      conv2d_group_update_cir_buf(input_channels, input_channels_pad,kernel_channels,input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state,0);

      // Update index to input width padded
      idx_beg_inp_width_pad += x_str;

      // Convolution using matXvec with matrix as circular buffer
      const WORD64 *p_bias_grp = NULL;
      if(p_bias != NULL){
        p_bias_grp = p_bias+grp_i*kernels_per_group;
      }      
      xa_nn_matXvec_sym8sxsym16s_sym16s_circ
        (tmp_out /* output */
        ,p_state->cir_buf.p_curr/* matrix: rows x cols */
        ,(p_kernel+grp_i*kernels_per_group*kernel_channels*kernel_width*kernel_height) /* vec: cols */
        ,p_bias_grp /* bias */
        ,out_h /* rows */
        ,kernel_channels * ker_w * ker_h /* cols */
        ,kernel_channels * ker_w * y_str/* row_offset */
        ,kernels_per_group /* vec_count */
        ,kernel_channels * ker_w * ker_h /* vec_stride */
        ,out_channels_offset /* out_col_offset */
        ,out_height_offset /* out_row_offset */
        ,input_zero_bias
        ,(p_out_multiplier+grp_i*kernels_per_group)
        ,(p_out_shift+grp_i*kernels_per_group)
        ,out_zero_bias
        ,out_activation_min
        ,out_activation_max
        ,NULL
        );
      tmp_out += out_width_offset;
    }
  }

  return 0;
}

WORD32 xa_nn_conv2d_per_chan_sym8sxsym16s(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD64* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
  return xa_nn_conv2d_v2_per_chan_sym8sxsym16s(
              p_out, p_inp, p_kernel, p_bias, input_height,
              input_width, input_channels, kernel_height,
              kernel_width, kernel_channels, dilation_height,
              dilation_width, out_channels, x_stride, y_stride,
              x_padding, y_padding, out_height, out_width,
              input_zero_bias, p_out_multiplier, p_out_shift,
              out_zero_bias, out_data_format, p_scratch,
              -32768, 32767, NULL);
}
