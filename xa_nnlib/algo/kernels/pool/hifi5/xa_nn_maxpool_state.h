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

#ifndef __XA_NN_MAXPOOL_STATE_H__
#define __XA_NN_MAXPOOL_STATE_H__

#define ALIGNMENT   16   /* 16 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))
#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define LIMIT(input, min, max) \
    input = XT_MAX(min, XT_MIN(max, input));

typedef struct _xa_nn_maxpool_state_t
{
    pVOID p_scratch;
} xa_nn_maxpool_state_t;

WORD32 xa_nn_maxpool_init(
    WORD32 inp_precision,
    VOID   *handle,
    WORD32 input_width,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 out_width);

void xa_nn_maxpool_f32_hwc(
      FLOAT32* __restrict__ p_out,
const FLOAT32* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in);

void xa_nn_maxpool_f16_hwc(
      WORD16* __restrict__ p_out,
const WORD16* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in);

void xa_nn_maxpool_asym8_hwc(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in);

void xa_nn_maxpool_8_hwc(
      WORD8* __restrict__ p_out,
const WORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in);

void xa_nn_maxpool_16_hwc(
      WORD16* __restrict__ p_out,
const WORD16* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in);

/* NHWC f16 max-pool with packed argmax indices, VALID padding only.
 * act_min_bits / act_max_bits : raw IEEE-754 fp16 bit patterns (WORD16).
 * ID encoding: p_out_Id[oh,ow,c] = (best_ky << 4) | (best_kx & 0xF) */
void xa_nn_maxpoolId_v2_f16_nhwc(
      WORD16  * __restrict__ p_out,
      UWORD8  * __restrict__ p_out_Id,
const WORD16  * __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   out_height,
      WORD32   out_width,
      WORD16   act_min_bits,
      WORD16   act_max_bits);

#endif /* #ifndef __XA_NN_MAXPOOL_STATE_H__ */
