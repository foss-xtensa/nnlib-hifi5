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

WORD32 xa_nn_conv2d_depthwise_getsize
  (WORD32 input_width
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 x_stride
   ,WORD32 y_stride
   ,WORD32 x_padding
   ,WORD32 output_width
   ,WORD32 circ_buf_precision
   )
{
  XA_NNLIB_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((x_stride <= 0 || x_stride > kernel_width), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0 || y_stride > kernel_height), -1);
  XA_NNLIB_CHK_COND((x_padding < 0), -1);
  XA_NNLIB_CHK_COND((output_width <= 0), -1);

  WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

  WORD32 scratch_bytewidth = 0;
  WORD32 circ_buf_bytewidth = 0;

  switch (circ_buf_precision)
  {
    case 8: /* For 8b */
    case 16: /* For 16b */
      scratch_bytewidth = 8; /* 64b scratch */
      circ_buf_bytewidth = (circ_buf_precision/8); /* bytewidth as per precision */
      break;

    case -1: /* For f32 */
      scratch_bytewidth = 4; /* f32 scratch */
      circ_buf_bytewidth = 4; /* bytewidth for f32 */
      break;

    case -3: /* For asym8 */
      scratch_bytewidth = 4;
      circ_buf_bytewidth = 1;
      break;

    default:
      return -1; /* Retunrning due to invalid input */
      break;
  }

  int total_size, state_size, circ_buf_size, scratch_size;
  int circ_buf_width;
  int output_width_for_x_stride_1;
  int output_height;
  state_size = ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_state_t), ALIGNMENT_16);
  circ_buf_size =
    xa_nn_circ_buf_getsize
    (circ_buf_bytewidth
     ,input_width
     ,kernel_height
     ,kernel_width
     ,x_stride
     ,y_stride
     ,x_padding
     ,circ_buf_height
     ,output_width
    );
  if (0 > circ_buf_size)
  {
    /* Returning negative error value as is to callee function to notify it.
     * Callee function should handle this negative value with care to avoid
     * any memory alloc issues. */
    return -1;
  }

  /* Get aligned size so as to have next memory pointer aligned */
  circ_buf_size = ALIGNED_SIZE(circ_buf_size, ALIGNMENT_16);

  circ_buf_width = kernel_width + ((output_width - 1) * x_stride);
  circ_buf_width = XT_MAX(circ_buf_width, x_padding+input_width);
  if(circ_buf_bytewidth == 1 || circ_buf_bytewidth == 2)
    circ_buf_width = ALIGNED_SIZE(circ_buf_width, 8);
  else
    circ_buf_width = ALIGNED_SIZE(circ_buf_width, 4);

  /* Please note for future output_width_for_x_stride_1 calculation for getting output_width_for_x_stride_1
   * from circ_buf_width with stride 1 (for x direction) will be as follows.
   * */
  output_width_for_x_stride_1 = (1 + ((circ_buf_width - kernel_width)/1));

  /* output_width_for_x_stride_1 loop is unrolled by 4 so keeping this dimension to multiple of 4 */
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, 4);

  output_height = (1 + ((circ_buf_height - kernel_height) / (y_stride)));

  scratch_size = (output_height * output_width_for_x_stride_1 * scratch_bytewidth);
  /* Get aligned size so as to have next memory pointer aligned */
  scratch_size = ALIGNED_SIZE(scratch_size, ALIGNMENT_16);

  total_size = state_size + circ_buf_size + scratch_size;

  if (0 > total_size)
  {
    return -1;
  }
  else
  {
    return total_size;
  }
}

VOID xa_nn_conv2d_depthwise_init
  (pVOID p_scratch
   ,WORD32 input_width
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 x_stride
   ,WORD32 y_stride
   ,WORD32 x_padding
   ,WORD32 output_width
   ,WORD32 circ_buf_precision
   )

{
    WORD32 circ_buf_bytewidth = 0;

    switch (circ_buf_precision)
    {
        case 8: /* For 8b */
        case 16: /* For 16b */
            circ_buf_bytewidth = (circ_buf_precision/8);
        break;

        case -1: /* For f32 */
            circ_buf_bytewidth = 4;
        break;

        case -3: /* For asym8 */
            circ_buf_bytewidth = 1;

        default:
        break;
    }

    WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

    pWORD8 p_mem = p_scratch;
    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_mem;
    int state_size, circ_buf_size;
    state_size = sizeof(xa_nn_conv2d_dw_state_t);
    p_mem = (p_mem + state_size);
    p_mem = (pWORD8)ALIGN_PTR(p_mem, ALIGNMENT_16);
    xa_nn_circ_buf_init(&(p_state->circ_buf)
                        ,p_mem
                        ,circ_buf_bytewidth
                        ,input_width
                        ,kernel_height
                        ,kernel_width
                        ,x_stride
                        ,y_stride
                        ,x_padding
                        ,circ_buf_height
                        ,output_width
                        );

    circ_buf_size = (int)((unsigned)p_state->circ_buf.p_end - (unsigned)p_state->circ_buf.p_begin);
    /* Get aligned size so as to have next memory pointer aligned */
    circ_buf_size = circ_buf_size;

    /* Every row of circular buffer is 8 byte aligned so don't need ALIGNED_SIZE for circular
    buffer size */
    p_mem = (p_mem + circ_buf_size);
    p_mem = (pWORD8)ALIGN_PTR(p_mem, ALIGNMENT_16);
    p_state->p_scratch = (pVOID)p_mem;
}


