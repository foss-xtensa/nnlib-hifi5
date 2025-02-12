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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

static WORD32 xa_nn_conv2d_pointwise_v2_nhwc_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_kernel,
    WORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32*  __restrict__ p_out_multiplier,
    WORD32*  __restrict__ p_out_shift,
    WORD32  out_zero_bias,
    WORD32  out_activation_min,
    WORD32  out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  int ret, out_plane_size;
  out_plane_size = input_height*input_width;
  int vec_offset, out_offset;

  vec_offset = input_channels;
  out_offset = out_channels;

  int tile_size;
  {
    tile_size = out_plane_size;
    while(out_channels*input_channels + tile_size*input_channels + tile_size*out_channels + out_channels*4*3 + XCHAL_DCACHE_LINESIZE*6 > (XCHAL_DCACHE_SIZE>>1) && tile_size > 0)
    {
      tile_size -= 4;
    }
    tile_size = (tile_size + 3)&(~3);
    if(tile_size <= 0 || tile_size > out_plane_size)
      tile_size = out_plane_size;
  }
  int itr_t;
  for(itr_t = 0; itr_t < out_plane_size; )
  {
    ret = xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(&p_out[itr_t*out_channels],
                                          p_kernel,
                                          &p_inp[itr_t*input_channels],
                                          p_bias,
                                          out_channels,
                                          input_channels,
                                          input_channels,
                                          tile_size,
                                          vec_offset,
                                          out_offset,
                                          1,
                                          input_zero_bias,
                                          p_out_multiplier,
                                          p_out_shift,
                                          out_zero_bias,
                                          out_activation_min,
                                          out_activation_max,
                                          p_dma_cfg
                                          );
    itr_t += tile_size;
    tile_size = tile_size < out_plane_size - itr_t ? tile_size : out_plane_size - itr_t;
  }
  if(ret<0)
      return ret;
  return 0;
}


static WORD32 xa_nn_conv2d_pointwise_v2_nchw_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_kernel,
    WORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32* __restrict__ p_out_multiplier,
    WORD32* __restrict__ p_out_shift,
    WORD32  out_zero_bias,
    WORD32  out_activation_min,
    WORD32  out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  int ret, out_plane_size;
  out_plane_size = input_height*input_width;
  int vec_offset, out_offset;

  vec_offset = input_channels;
  out_offset = 1;

  ret = xa_nn_matmul_v2_per_chan_sym8sxasym8s_asym8s(p_out,
                                        p_kernel,
                                        p_inp,
                                        p_bias,
                                        out_channels,
                                        input_channels,
                                        input_channels,
                                        out_plane_size,
                                        vec_offset,
                                        out_offset,
                                        out_plane_size,
                                        input_zero_bias,
                                        p_out_multiplier,
                                        p_out_shift,
                                        out_zero_bias,
                                        out_activation_min,
                                        out_activation_max,
                                        p_dma_cfg
                                        );
  if(ret<0)
      return ret;
  return 0;
}

WORD32 xa_nn_conv2d_pointwise_v2_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_kernel,
    WORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32* __restrict__ p_out_multiplier,
    WORD32* __restrict__ p_out_shift,
    WORD32  out_zero_bias,
    WORD32  out_data_format,
    WORD32  out_activation_min,
    WORD32  out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias > 127 || out_zero_bias < -128), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min || out_activation_max > 127), -1);

  int itr = 0;
  for(itr=0; itr<out_channels; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }
  int ret = 0;

  if(out_data_format == 0){
    ret = xa_nn_conv2d_pointwise_v2_nhwc_per_chan_sym8sxasym8s(
          p_out,
          p_kernel,
          p_inp,
          p_bias,
          input_height,
          input_width,
          input_channels,
          out_channels,
          input_zero_bias,
          p_out_multiplier,
          p_out_shift,
          out_zero_bias,
          out_activation_min,
          out_activation_max,
          p_dma_cfg);
  }
  else if(out_data_format == 1){
    ret = xa_nn_conv2d_pointwise_v2_nchw_per_chan_sym8sxasym8s(
          p_out,
          p_kernel,
          p_inp,
          p_bias,
          input_height,
          input_width,
          input_channels,
          out_channels,
          input_zero_bias,
          p_out_multiplier,
          p_out_shift,
          out_zero_bias,
          out_activation_min,
          out_activation_max,
          p_dma_cfg);
  }
  return ret;
}

WORD32 xa_nn_conv2d_pointwise_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_kernel,
    WORD8* __restrict__ p_inp,
    WORD32* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  input_zero_bias,
    WORD32* __restrict__ p_out_multiplier,
    WORD32* __restrict__ p_out_shift,
    WORD32  out_zero_bias,
    WORD32  out_data_format)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias > 127 || out_zero_bias < -128), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr = 0;
  for(itr=0; itr<out_channels; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }
  int ret = 0;

  if(out_data_format == 0){
    ret = xa_nn_conv2d_pointwise_v2_nhwc_per_chan_sym8sxasym8s(
          p_out,
          p_kernel,
          p_inp,
          p_bias,
          input_height,
          input_width,
          input_channels,
          out_channels,
          input_zero_bias,
          p_out_multiplier,
          p_out_shift,
          out_zero_bias,
          -128,
          127,
          NULL);
  }
  else if(out_data_format == 1){
    ret = xa_nn_conv2d_pointwise_v2_nchw_per_chan_sym8sxasym8s(
          p_out,
          p_kernel,
          p_inp,
          p_bias,
          input_height,
          input_width,
          input_channels,
          out_channels,
          input_zero_bias,
          p_out_multiplier,
          p_out_shift,
          out_zero_bias,
          -128,
          127,
          NULL);
  }
  return ret;
}
