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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

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
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  WORD32 left_shift, right_shift;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

#if TFLITE_SINGLE_ROUNDING
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#endif

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift = p_out_shift[k];
#if XCHAL_HAVE_HIFI5S
        left_shift = 31 - left_shift;
        left_shift = (left_shift << 16) | left_shift;
#endif        
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#else
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#endif        
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc, min_int8, max_int8);
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
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
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

#if TFLITE_SINGLE_ROUNDING
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#endif

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift = p_out_shift[k];
#if XCHAL_HAVE_HIFI5S
        left_shift = 31 - left_shift;
        left_shift = (left_shift << 16) | left_shift;
#endif          
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#endif        
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc, min_int8, max_int8);
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
      }
    }
  }
  return out_width_over_x_r_pad;
}


static void conv_y_pad_nhwc_out(
    WORD8 *__restrict__ p_out,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 out_channels,
    const WORD32* __restrict__ p_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 left_shift, right_shift;
  WORD32 out_height_offset, out_width_offset, out_channels_offset;

  out_channels_offset = 1;
  out_width_offset = out_channels;
  out_height_offset = out_width * out_width_offset;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

#if TFLITE_SINGLE_ROUNDING
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#endif

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift = p_out_shift[k];
#if XCHAL_HAVE_HIFI5S
        left_shift = 31 - left_shift;
        left_shift = (left_shift << 16) | left_shift;
#endif         
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
#endif        
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc, min_int8, max_int8);
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (WORD8)AE_MOVAD32_L(acc);
      }
    }
  }
}

__attribute__ ((noinline)) 
static WORD32 internal_xa_nn_conv2d_std_per_chan_sym4sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
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
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv2d_std_init_state_sym4s((void*)p_state,(void*)p_kernel,input_height,input_channels,kernel_height,kernel_width,x_stride,y_stride,y_padding,out_height,out_channels,-4);

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;
  WORD32 input_channels_pad = input_channels;

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= kernel_width)
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_stride;

    // Convolution using matXvec with matrix as circular buffer

    xa_nn_matXvec_sym4sxasym8s_asym8s_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_state->p_kernel_padded /* vec: cols */
       ,p_bias /* bias */
       ,out_height /* rows */
       ,input_channels_pad * kernel_width * kernel_height /* cols */
       ,input_channels_pad * kernel_width * y_stride/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,input_zero_bias
       ,p_out_multiplier
       ,p_out_shift
       ,out_zero_bias
      );

    p_out += out_width_offset;
  }

  return 0;
}


WORD32 xa_nn_conv2d_std_per_chan_sym4sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
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
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((((input_channels * kernel_width * kernel_height) % 2) != 0), -1);
  
  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }
  
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

  int ret = 0;
  int tile_height = out_h;
  int mem_req = 0;

  do
  {
    mem_req = tile_height * out_w * out_channels + ker_h * ker_w * input_channels;
    mem_req += ((tile_height - 1)*y_str + ker_h) * inp_w * input_channels;
    mem_req += xa_nn_conv2d_std_getsize(((tile_height - 1)*y_str + ker_h),inp_w, input_channels,
                    ker_h, ker_w, input_channels, y_str, y_pad, x_str, x_pad, tile_height, out_w, out_channels, PREC_ASYM8S, PREC_SYM4S,1,1,out_data_format);
    mem_req += tile_height * 3 * sizeof(WORD32);
    if(mem_req < XCHAL_DCACHE_SIZE || tile_height <= 0)
      break;
    tile_height = (tile_height - 8) & (~3);
  } while(1);

  if(tile_height <= 0)
    tile_height = out_h;

  if(tile_height == out_h || out_data_format == 1)
  {
    ret |= internal_xa_nn_conv2d_std_per_chan_sym4sxasym8s(
        p_out,
        p_inp,
        p_kernel,
        p_bias,
        inp_h,
        inp_w,
        input_channels,
        ker_h,
        ker_w,
        out_channels,
        x_str,
        y_str,
        x_pad,
        y_pad,
        out_h,
        out_w,
        input_zero_bias,
        p_out_multiplier,
        p_out_shift,
        out_zero_bias,
        out_data_format,
        p_scratch);
  }
  else
  {
    int itr_oh, itr_ih;
    itr_oh = 0;
    itr_ih = -y_pad;
    do
    {
      int inp_h_idx = itr_ih < 0 ? 0 : (itr_ih >= inp_h ? inp_h - 1 : itr_ih);
      int y_padding_cur = itr_ih < 0 ? -itr_ih : 0;
      int inp_height_cur = (tile_height - 1)*y_str + ker_h - y_padding_cur;
      inp_height_cur = inp_height_cur > inp_h - inp_h_idx ? inp_h - inp_h_idx : inp_height_cur;
      inp_height_cur = itr_ih >= inp_h ? 0 : inp_height_cur;

      if(inp_height_cur == 0)
      {
        conv_y_pad_nhwc_out(
            &p_out[itr_oh * out_w * out_channels],
            tile_height,
            out_w,
            out_channels,
            p_bias,
            p_out_multiplier,
            p_out_shift,
            out_zero_bias);
      }
      else
      {
        ret |= internal_xa_nn_conv2d_std_per_chan_sym4sxasym8s(
            &p_out[itr_oh * out_w * out_channels],
            &p_inp[inp_h_idx * inp_w * input_channels],
            p_kernel,
            p_bias,
            inp_height_cur,
            inp_w,
            input_channels,
            ker_h,
            ker_w,
            out_channels,
            x_str,
            y_str,
            x_pad,
            y_padding_cur,
            tile_height,
            out_w,
            input_zero_bias,
            p_out_multiplier,
            p_out_shift,
            out_zero_bias,
            out_data_format,
            p_scratch);
      }
      itr_oh += tile_height;
      itr_ih += tile_height * y_str;
      tile_height = tile_height > out_h - itr_oh ? out_h - itr_oh : tile_height;
    } while(itr_oh < out_h);
  }

  return ret;
}
