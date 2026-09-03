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
#include <string.h>
#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_std_f16,(
    WORD16 *p_out,
    const WORD16 *p_inp,
    const WORD16 *p_kernel,
    const WORD16 *p_bias,
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
    WORD32 out_data_format,
    VOID *p_handle))
  DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_std_v2_f16,(
    WORD16 *p_out,
    const WORD16 *p_inp,
    const WORD16 *p_kernel,
    const WORD16 *p_bias,
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
    WORD32 out_data_format,
    VOID *p_handle,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    struct _xa_dma_cfg_t *p_dma_cfg))
#else /* #if !HAVE_HP_VFPU */

typedef xthalf FLOAT16;

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
    FLOAT16 *p_bias,
    FLOAT16 act_min_val,
    FLOAT16 act_max_val,
    FLOAT16 *p_out)
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
        FLOAT16 out_val = p_bias[k];
        out_val = MAX_H(MIN_H(out_val, act_max_val), act_min_val);
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] = out_val;
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
    FLOAT16 *p_bias,
    FLOAT16 act_min_val,
    FLOAT16 act_max_val,
    FLOAT16 *p_out)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i=0;i<out_height;i++)
  {
    for(j=idx_out_width_over_x_r_pad;j<out_width;j++)
    {
      for(k=0;k<out_channels;k++)
      {
        FLOAT16 out_val = p_bias[k];
        out_val = MAX_H(MIN_H(out_val, act_max_val), act_min_val);
        p_out[i*out_height_offset+j*out_width_offset+k*out_channels_offset] = out_val;
      }
    }
  }
  return out_width_over_x_r_pad;
}

static WORD32 gcd_f16(WORD32 a, WORD32 b)
{
  while(b != 0)
  {
    WORD32 temp = a % b;
    a = b;
    b = temp;
  }
  return a;
}

static WORD32 f16_bits_less_than(WORD16 lhs_bits, WORD16 rhs_bits)
{
  UWORD16 lhs = (UWORD16)lhs_bits;
  UWORD16 rhs = (UWORD16)rhs_bits;
  UWORD16 lhs_key, rhs_key;

  if((((lhs & 0x7C00u) == 0x7C00u) && (lhs & 0x03FFu)) ||
     (((rhs & 0x7C00u) == 0x7C00u) && (rhs & 0x03FFu)))
  {
    return 0;
  }

  if((lhs & 0x7FFFu) == 0)
  {
    lhs = 0;
  }
  if((rhs & 0x7FFFu) == 0)
  {
    rhs = 0;
  }

  lhs_key = (lhs & 0x8000u) ? (UWORD16)(~lhs) : (UWORD16)(lhs ^ 0x8000u);
  rhs_key = (rhs & 0x8000u) ? (UWORD16)(~rhs) : (UWORD16)(rhs ^ 0x8000u);

  return lhs_key < rhs_key;
}

static WORD32 xa_nn_conv2d_std_f16_internal(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_bias,
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
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD16 act_min_bits,
    WORD16 act_max_bits)
{
  FLOAT16 act_min_val = *((FLOAT16 *)&act_min_bits);
  FLOAT16 act_max_val = *((FLOAT16 *)&act_max_bits);

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT16), -1);
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
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND(f16_bits_less_than(act_max_bits, act_min_bits), -1);
  
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
  WORD32 input_bytewidth = sizeof(*p_inp);
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv2d_std_init_state((void*)p_state,(void*)p_kernel,inp_h,input_channels,ker_h,ker_w,out_channels,x_str,y_str,y_pad,out_h,-2);

  WORD32 out_channels_offset = out_data_format ? out_h * out_w : 1;
  WORD32 out_height_offset = out_data_format ? out_w : out_w * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_pad;
  WORD32 input_channels_pad = input_channels;

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, (FLOAT16 *)p_bias, act_min_val, act_max_val, (FLOAT16 *)p_out);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, (FLOAT16 *)p_bias, act_min_val, act_max_val, (FLOAT16 *)p_out);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = ker_w - x_str;

  // Process Loop to compute one output plane [out_h x out_channels] per iteration
  for(j=0;j<out_w-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_str x (inp_h x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf(input_channels, input_channels_pad, input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_str;

    // calculate padded kernel size
    int per_kernel_size = kernel_height * kernel_width * input_channels;
    // per_kernel_size = PADDED_SIZE(per_kernel_size, 8);


    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_f16_circ
      ((FLOAT16 *)p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_state->p_kernel_padded /* vec: cols */
       ,(FLOAT16 *)p_bias /* bias */
       ,out_h /* rows */
       ,per_kernel_size /* cols */
       ,input_channels_pad * ker_w * y_str/* row_offset */
       ,out_channels /* vec_count */
       ,PADDED_SIZE(per_kernel_size, 8) /* vec_offset */
       ,out_channels_offset /* out_col_offset */
      ,out_height_offset /* out_row_offset */
      ,act_min_bits
      ,act_max_bits
      );

    p_out += out_width_offset;
  }

  return 0;
}

WORD32 xa_nn_conv2d_std_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_bias,
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
    WORD32 out_data_format,
    VOID *p_scratch)
{
  return xa_nn_conv2d_std_f16_internal(
      p_out, p_inp, p_kernel, p_bias,
      input_height, input_width, input_channels,
      kernel_height, kernel_width, out_channels,
      x_stride, y_stride, x_padding, y_padding,
      out_height, out_width, out_data_format, p_scratch,
      (WORD16)0xFC00u, (WORD16)0x7C00u);
}

WORD32 xa_nn_conv2d_std_v2_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_bias,
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
    WORD32 out_data_format,
    VOID *p_scratch,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    struct _xa_dma_cfg_t *p_dma_cfg)
{
  WORD16 act_min_bits = (out_activation_min != NULL) ? *out_activation_min : (WORD16)0xFC00u;
  WORD16 act_max_bits = (out_activation_max != NULL) ? *out_activation_max : (WORD16)0x7C00u;

  (void)p_dma_cfg;

  return xa_nn_conv2d_std_f16_internal(
      p_out, p_inp, p_kernel, p_bias,
      input_height, input_width, input_channels,
      kernel_height, kernel_width, out_channels,
      x_stride, y_stride, x_padding, y_padding,
      out_height, out_width, out_data_format, p_scratch,
      act_min_bits, act_max_bits);
}

WORD32 xa_nn_dilated_conv2d_std_v2_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 inp_data_format,
    WORD32 out_data_format,
    pVOID p_scratch,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    struct _xa_dma_cfg_t *p_dma_cfg)
{
  WORD16 *p_out_base = p_out;
  WORD16 act_min_bits = (out_activation_min != NULL) ? *out_activation_min : (WORD16)0xFC00u;
  WORD16 act_max_bits = (out_activation_max != NULL) ? *out_activation_max : (WORD16)0x7C00u;
  FLOAT16 act_min_val = *((FLOAT16 *)&act_min_bits);
  FLOAT16 act_max_val = *((FLOAT16 *)&act_max_bits);
  WORD32 kernel_height_dilation;
  WORD32 kernel_width_dilation;
  WORD32 input_bytewidth = sizeof(FLOAT16);
  VOID *pp_inp = (VOID *)p_inp;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 out_channels_offset;
  WORD32 out_height_offset;
  WORD32 out_width_offset;
  WORD32 x_padding_var;
  WORD32 input_channels_pad = input_channels;
  WORD32 out_width_over_x_pad = 0;
  WORD32 out_width_over_x_r_pad = 0;
  WORD32 x_r_pad;
  WORD32 y_b_pad;
  WORD32 per_kernel_size;
  WORD32 per_kernel_size_padded;
  WORD32 dilation_w_offset;
  WORD32 dilation_h_offset;
  WORD32 out_iteraions;

  (VOID)p_dma_cfg;
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);

  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);

  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((x_stride <= 0 || y_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((x_padding < 0 || y_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);

  if(kernel_height == 1)
  {
    dilation_height = 1;
  }
  if(kernel_width == 1)
  {
    dilation_width = 1;
  }

  kernel_height_dilation = kernel_height + ((dilation_height - 1) * (kernel_height - 1));
  kernel_width_dilation = kernel_width + ((dilation_width - 1) * (kernel_width - 1));

  out_channels_offset = 1;
  out_height_offset = out_width * out_channels;
  out_width_offset = out_channels;
  x_padding_var = x_padding;
  per_kernel_size = kernel_height * kernel_width * input_channels;
  per_kernel_size_padded = PADDED_SIZE(per_kernel_size, 8);

  xa_nn_conv2d_dilation_init_state((VOID *)p_state, (VOID *)p_kernel, pp_inp);
  xa_nn_dilated_conv2d_std_init_circ_buf(
      (VOID *)p_state,
      (VOID *)p_kernel,
      input_height,
      input_channels,
      kernel_height_dilation,
      kernel_width,
      x_stride,
      y_stride,
      y_padding,
      out_height,
      -2,
      dilation_height,
      0);

  p_state->p_kernel_padded = (VOID *)p_kernel;
  if(per_kernel_size != per_kernel_size_padded)
  {
    FLOAT16 *p_kernel_padded = (FLOAT16 *)ALIGNED_ADDR(p_state->cir_buf.p_end, ALIGNMENT_16);
    const FLOAT16 *p_kernel_src = (const FLOAT16 *)p_kernel;
    WORD32 out_channel;

    p_state->p_kernel_padded = (VOID *)p_kernel_padded;
    for(out_channel = 0; out_channel < out_channels; out_channel++)
    {
      memcpy(p_kernel_padded, p_kernel_src, per_kernel_size * sizeof(FLOAT16));
      memset(p_kernel_padded + per_kernel_size, 0, (per_kernel_size_padded - per_kernel_size) * sizeof(FLOAT16));
      p_kernel_padded += per_kernel_size_padded;
      p_kernel_src += per_kernel_size;
    }
  }

  if(x_padding_var >= kernel_width_dilation)
  {
    out_width_over_x_pad = conv_x_left_pad(
        x_padding,
        kernel_width_dilation,
        x_stride,
        out_width,
        out_height,
        out_channels,
        out_channels_offset,
        out_width_offset,
        out_height_offset,
        (FLOAT16 *)p_bias,
        act_min_val,
        act_max_val,
        (FLOAT16 *)p_out);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  x_r_pad = kernel_width_dilation + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width_dilation)
  {
    out_width_over_x_r_pad = conv_x_right_pad(
        x_padding,
        input_width,
        x_stride,
        out_width,
        out_height,
        out_channels,
        out_channels_offset,
        out_width_offset,
        out_height_offset,
        (FLOAT16 *)p_bias,
          act_min_val,
          act_max_val,
        (FLOAT16 *)p_out);
  }

  y_b_pad = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > (y_padding + input_height + y_b_pad)), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width_dilation > (x_padding + input_width + x_r_pad)), -1);

  for(dilation_w_offset = 0; dilation_w_offset < dilation_width; dilation_w_offset++)
  {
    WORD32 x_padding_dilation_initial_pad = ((x_padding - x_padding_var) / dilation_width) +
        (WORD32)((((x_padding - x_padding_var) % dilation_width) - 1) >= dilation_w_offset);
    WORD32 x_stride_dilated = x_stride / gcd_f16(x_stride, dilation_width);
    WORD32 widthIndexIteration;
    WORD32 firstWidthIndexNr;
    WORD32 firstWidthIndex;
    WORD32 adjustZpAndOffsetIndex;
    WORD32 totalPointsParticipatingInConvolution;
    WORD32 pointsParticipatingInConvolutionForThisOffset;
    WORD32 out_points_for_this_xyoffset;

    for(widthIndexIteration = 0; widthIndexIteration < x_stride_dilated; widthIndexIteration++)
    {
      firstWidthIndexNr = dilation_w_offset + (widthIndexIteration * dilation_width);
      firstWidthIndex = firstWidthIndexNr / x_stride;
      if((firstWidthIndex * x_stride) == firstWidthIndexNr)
      {
        break;
      }
    }
    if(widthIndexIteration == x_stride_dilated)
    {
      continue;
    }

    if(x_padding_dilation_initial_pad <= widthIndexIteration)
    {
      adjustZpAndOffsetIndex = widthIndexIteration - x_padding_dilation_initial_pad;
    }
    else
    {
      adjustZpAndOffsetIndex = (x_padding_dilation_initial_pad - widthIndexIteration) / x_stride_dilated;
      adjustZpAndOffsetIndex +=
          (((x_padding_dilation_initial_pad - widthIndexIteration) -
            (adjustZpAndOffsetIndex * x_stride_dilated)) > 0);
      adjustZpAndOffsetIndex = widthIndexIteration + (adjustZpAndOffsetIndex * x_stride_dilated);
      adjustZpAndOffsetIndex -= x_padding_dilation_initial_pad;
    }

    totalPointsParticipatingInConvolution = x_padding + input_width +
        (x_r_pad - (out_width_over_x_r_pad * x_stride));
    pointsParticipatingInConvolutionForThisOffset =
        (totalPointsParticipatingInConvolution / dilation_width) +
        (WORD32)((((totalPointsParticipatingInConvolution % dilation_width) - 1) >= dilation_w_offset));
    pointsParticipatingInConvolutionForThisOffset -= x_padding_dilation_initial_pad;

    if((pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) < kernel_width)
    {
      continue;
    }

    out_points_for_this_xyoffset =
        ((pointsParticipatingInConvolutionForThisOffset - adjustZpAndOffsetIndex) - kernel_width) /
            x_stride_dilated +
        1;

    for(dilation_h_offset = 0; dilation_h_offset < dilation_height; dilation_h_offset++)
    {
      WORD32 input_padding_consumed = 0;
      WORD32 input_width_consumed = 0;
      WORD32 y_stride_dilated = y_stride / gcd_f16(y_stride, dilation_height);
      WORD32 heightIndexIteration;
      WORD32 firstHeightIndexNr;
      WORD32 firstHeightIndex;
      WORD32 heightOfCircMatrix;
      WORD32 circMatrixHeight = 0;

      for(heightIndexIteration = 0; heightIndexIteration < y_stride_dilated; heightIndexIteration++)
      {
        firstHeightIndexNr = dilation_h_offset + (heightIndexIteration * dilation_height);
        firstHeightIndex = firstHeightIndexNr / y_stride;
        if((firstHeightIndex * y_stride) == firstHeightIndexNr)
        {
          break;
        }
      }

      heightOfCircMatrix = ((y_padding + input_height + y_b_pad) / dilation_height) +
          (WORD32)((((y_padding + input_height + y_b_pad) % dilation_height) - 1) >= dilation_h_offset);
      if(heightIndexIteration == y_stride_dilated)
      {
        continue;
      }
      if((heightOfCircMatrix - heightIndexIteration) < kernel_height)
      {
        continue;
      }

      xa_nn_dilated_conv2d_std_init_circ_buf(
          (VOID *)p_state,
          (VOID *)p_kernel,
          input_height,
          input_channels,
          kernel_height_dilation,
          kernel_width,
          x_stride,
          y_stride,
          y_padding,
          out_height,
          -2,
          dilation_height,
          dilation_h_offset);

      {
        WORD32 planesToAdd = XT_MAX(kernel_width - x_stride_dilated, 0);
        WORD32 outPointerHeightOffset;
        WORD32 outPointerWidthOffset;

        xa_nn_dilated_conv2d_std_load_cir_buf_asym8(
            input_channels,
            input_channels_pad,
            input_bytewidth,
            input_width,
            input_height,
            y_padding,
            y_b_pad,
            x_padding_var,
            kernel_width,
            x_stride,
            (VOID **)&pp_inp,
            p_state,
            0,
            dilation_height,
            dilation_h_offset,
            dilation_width,
            dilation_w_offset,
            x_padding,
            &input_padding_consumed,
            &input_width_consumed,
            planesToAdd,
            2,
            &circMatrixHeight,
            adjustZpAndOffsetIndex,
            x_stride_dilated,
            heightIndexIteration,
            y_stride_dilated);

        outPointerHeightOffset = (dilation_h_offset + (heightIndexIteration * dilation_height)) / y_stride;
        outPointerWidthOffset =
            (((x_padding_dilation_initial_pad + adjustZpAndOffsetIndex) * dilation_width) +
             dilation_w_offset) /
            x_stride;
        p_out = p_out_base + (outPointerHeightOffset * out_height_offset) +
            (outPointerWidthOffset * out_width_offset);

        for(out_iteraions = 0; out_iteraions < out_points_for_this_xyoffset; out_iteraions++)
        {
          WORD32 planesToAddIter = XT_MIN(x_stride_dilated, kernel_width);

          xa_nn_dilated_conv2d_std_load_cir_buf_asym8(
              input_channels,
              input_channels_pad,
              input_bytewidth,
              input_width,
              input_height,
              y_padding,
              y_b_pad,
              x_padding_var,
              kernel_width,
              x_stride,
              (VOID **)&pp_inp,
              p_state,
              0,
              dilation_height,
              dilation_h_offset,
              dilation_width,
              dilation_w_offset,
              x_padding,
              &input_padding_consumed,
              &input_width_consumed,
              planesToAddIter,
              0,
              &circMatrixHeight,
              adjustZpAndOffsetIndex,
              x_stride_dilated,
              heightIndexIteration,
              y_stride_dilated);

          xa_nn_matXvec_f16_circ(
              (FLOAT16 *)p_out,
              (FLOAT16 *)p_state->cir_buf.p_curr,
              (FLOAT16 *)p_state->p_kernel_padded,
              (FLOAT16 *)p_bias,
              ((circMatrixHeight - kernel_height) / y_stride_dilated) + 1,
              input_channels_pad * kernel_width * kernel_height,
              input_channels_pad * kernel_width * y_stride_dilated,
              out_channels,
              per_kernel_size_padded,
              out_channels_offset,
              out_height_offset * dilation_height / gcd_f16(y_stride, dilation_height),
              act_min_bits,
              act_max_bits);

          p_out += out_width_offset * dilation_width / gcd_f16(x_stride, dilation_width);
        }
      }
    }
  }

  return 0;
}
#endif /* #if !HAVE_VFPU */

