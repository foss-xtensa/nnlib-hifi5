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

#ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__
#define __XA_NN_CONV2D_DEPTHWISE_STATE_H__

#include "xa_nn_circ_buf.h"

typedef struct _xa_nn_conv2d_dw_state_t
{
    xa_nn_circ_buf_t circ_buf;
    pVOID p_scratch;
} xa_nn_conv2d_dw_state_t;

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
typedef struct _xa_nn_conv2d_dw_k3x3_state_t
{
  /* Scratch buffer pointers */
  WORD8 *p_dummy_inp;
  WORD8 *p_kernel_rearranged;
  WORD8 *p_kernel_nchw;
  WORD32 *p_accu;
  WORD32 *p_accu_zero_point;
  WORD32 *p_scale_multipliers;

  /* Output height loop counter */
  int top_padded_region_output;
  int bottom_padded_region_output;
  int top_single_input_row_output;
  int middle_single_input_row_output;
  int top_two_input_row_output;
  int six_input_row_output;
  int three_input_row_output;
  int bottom_single_input_row_output;
  int bottom_two_input_row_output;

  const WORD8 *p_inp0;
  const WORD8 *p_inp1;
  const WORD8 *p_inp2;
  int inp0_offset, inp1_offset, inp2_offset;
} xa_nn_conv2d_dw_k3x3_state_t;
#endif

VOID xa_nn_dilated_conv2d_depthwise_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 dilation_height
 ,WORD32 dilation_width
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 ,pVOID p_pad_val
 );

#endif /* #ifndef __XA_NN_CONV2D_DEPTHWISE_STATE_H__ */
