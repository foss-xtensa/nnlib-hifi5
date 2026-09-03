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
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_depthwise_state.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_pointwise_v2_f16,(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
)
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_pointwise_f16,(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format))
#else /* #if !HAVE_HP_VFPU */

static WORD32 xa_nn_conv2d_pointwise_nhwc_f16(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = out_channels;

    ret = xa_nn_matmul_v2_f16xf16_f16(p_out,
                                   p_kernel,
                                   p_inp,
                                   p_bias,
                                   out_channels,
                                   input_channels,
                                   input_channels,
                                   out_plane_size,
                                   vec_offset,
                                   out_offset,
                                   1,
                                   out_activation_min,
                                   out_activation_max,
                                   p_dma_cfg
                                   );
    if(ret<0)
        return ret;
    return 0;
}

static WORD32 xa_nn_conv2d_pointwise_nchw_f16(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    int ret, out_plane_size;
    out_plane_size = input_height*input_width;
    int vec_offset, out_offset;

    vec_offset = input_channels;
    out_offset = 1;

    ret = xa_nn_matmul_v2_f16xf16_f16(p_out,
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
                                   out_activation_min,
                                   out_activation_max,
                                   p_dma_cfg
                                   );
    if(ret<0)
        return ret;
    return 0;
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

WORD32 xa_nn_conv2d_pointwise_v2_f16(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    WORD16 act_min_bits = (out_activation_min != NULL) ? *out_activation_min : (WORD16)0xFC00u;
    WORD16 act_max_bits = (out_activation_max != NULL) ? *out_activation_max : (WORD16)0x7C00u;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND(f16_bits_less_than(act_max_bits, act_min_bits), -1);
    

    int ret=0;

    if(out_data_format == 0){
        ret = xa_nn_conv2d_pointwise_nhwc_f16(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                out_channels,
                &act_min_bits,
                &act_max_bits,
                p_dma_cfg);
    }
    else if(out_data_format == 1){
        ret = xa_nn_conv2d_pointwise_nchw_f16(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                out_channels,
                &act_min_bits,
                &act_max_bits,
                p_dma_cfg);
    }
    return ret;
}

WORD32 xa_nn_conv2d_pointwise_f16(
    WORD16* __restrict__ p_out,
    WORD16* __restrict__ p_kernel,
    WORD16* __restrict__ p_inp,
    WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  out_channels,
    WORD32  out_data_format)
{
    return xa_nn_conv2d_pointwise_v2_f16(
        p_out,
        p_kernel,
        p_inp,
        p_bias,
        input_height,
        input_width,
        input_channels,
        out_channels,
        out_data_format,
        NULL,
        NULL,
        NULL);
}
#endif /* #if !HAVE_VFPU */
