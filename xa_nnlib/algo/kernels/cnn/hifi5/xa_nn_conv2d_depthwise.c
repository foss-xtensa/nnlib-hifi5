/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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

#ifdef AE_MULZB3X3O8X8
#define DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
#endif

static WORD32 xa_nn_conv2d_depthwise_nchw_getsize
  (WORD32 input_width
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 x_stride
   ,WORD32 y_stride
   ,WORD32 x_padding
   ,WORD32 output_width
   ,WORD32 circ_buf_bytewidth
   ,WORD32 scratch_bytewidth
   )
{
  int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
  int kernel_height_pad = ALIGNED_SIZE(kernel_height, 4);
  WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

  int total_size, state_size, circ_buf_size, scratch_size, padded_kernel_size;
  int circ_buf_width;
  int output_width_for_x_stride_1;
  int output_height;
  state_size = ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_state_t), ALIGNMENT_16);
  circ_buf_size =
    xa_nn_circ_buf_nchw_getsize
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

  /* output_width_for_x_stride_1 loop is unrolled by 8 so keeping this dimension to multiple of 8 */
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, 8);

  output_height = (1 + ((circ_buf_height - kernel_height) / (y_stride)));

  scratch_size = (output_height * output_width_for_x_stride_1 * scratch_bytewidth); //TODO: correct size
  /* Get aligned size so as to have next memory pointer aligned */
  scratch_size = ALIGNED_SIZE(scratch_size, ALIGNMENT_16);

  /* TBD: Exact calculation needs API change, using input bytewidth for now */
  padded_kernel_size = kernel_width_pad * kernel_height_pad * circ_buf_bytewidth;
  total_size = state_size + circ_buf_size + padded_kernel_size + scratch_size;

  if (0 > total_size)
  {
    return -1;
  }
  else
  {
    return total_size;
  }
}

static VOID xa_nn_conv2d_depthwise_nchw_init
  (pVOID p_scratch
   ,WORD32 input_width
   ,WORD32 kernel_height
   ,WORD32 kernel_width
   ,WORD32 x_stride
   ,WORD32 y_stride
   ,WORD32 x_padding
   ,WORD32 output_width
   ,WORD32 circ_buf_bytewidth
   ,pVOID p_pad_val
   )
{
    WORD32 circ_buf_height = (kernel_height + ((OUT_HEIGHT_PER_ITER - 1) * y_stride));

    pWORD8 p_mem = p_scratch;
    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_mem;
    int state_size, circ_buf_size;
    state_size = sizeof(xa_nn_conv2d_dw_state_t);
    p_mem = (p_mem + state_size);
    p_mem = (pWORD8)ALIGN_PTR(p_mem, ALIGNMENT_16);
    xa_nn_circ_buf_nchw_init(&(p_state->circ_buf)
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
                        ,p_pad_val
                        );

    circ_buf_size = (int)((unsigned)p_state->circ_buf.p_end - (unsigned)p_state->circ_buf.p_begin);
    /* Get aligned size so as to have next memory pointer aligned */

    /* Every row of circular buffer is 8 byte aligned so don't need ALIGNED_SIZE for circular
    buffer size */
    p_mem = (p_mem + circ_buf_size);
    p_mem = (pWORD8)ALIGN_PTR(p_mem, ALIGNMENT_16);
    p_state->p_scratch = (pVOID)p_mem;
}

static WORD32 xa_nn_conv2d_depthwise_nhwc_getsize
(WORD32 input_height
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 y_stride
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 circ_buf_bytewidth
 )
{
    int total_size, state_size, circ_buf_size;
#ifndef AE_MULZB3X3O8X8
    state_size = ALIGNED_SIZE(sizeof(xa_nn_circ_buf_t), ALIGNMENT);
#else
    state_size = ALIGNED_SIZE(sizeof(xa_nn_circ_buf_t), ALIGNMENT_16);
#endif
    circ_buf_size =
        xa_nn_circ_buf_nhwc_getsize
        (circ_buf_bytewidth
         ,input_height
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,y_stride
         ,y_padding
         ,output_height
        );

#ifndef AE_MULZB3X3O8X8
    if (0 > circ_buf_size)
    {
        return -1;
    }
    else
    {
        total_size = state_size + circ_buf_size;
        return total_size;
    }
#else
    if(circ_buf_bytewidth == 1)
    {
      int kernel_size, out_channels_pad;
      out_channels_pad = (input_channels * channels_multiplier + 15) & (~15);
      kernel_size = kernel_height * kernel_width * out_channels_pad;

      if (0 > circ_buf_size || 0 > kernel_size)
      {
          return -1;
      }
      else
      {
          total_size = state_size + circ_buf_size + kernel_size;
          return total_size;
      }
    }
    else
    {
      if (0 > circ_buf_size)
      {
          return -1;
      }
      else
      {
          total_size = state_size + circ_buf_size;
          return total_size;
      }
    }
#endif
}

#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE

#define KH_3X3 3
#define KW_3X3 3

static WORD32 xa_nn_conv2d_depthwise_getsize_k3x3
  (WORD32 input_height
  ,WORD32 input_width
  ,WORD32 input_channels
  ,WORD32 kernel_height
  ,WORD32 kernel_width
  )
{
  int total_size = 0;
  /* Alignment */
  total_size += ALIGNMENT_16;
  /* Handle */
  total_size += ALIGNED_SIZE(sizeof(xa_nn_conv2d_dw_k3x3_state_t), ALIGNMENT_16);
  /* Initial accumulator values: output bias */
  total_size += ALIGNED_SIZE(input_channels * sizeof(WORD32), ALIGNMENT_16);
  /* Initial accumulator values: output bias + zero_point adjustment */
  total_size += ALIGNED_SIZE(input_channels * sizeof(WORD32), ALIGNMENT_16);
  /* Output multipliers: l_mult, out_multiplier, r_mult */
  total_size += ALIGNED_SIZE(3 * input_channels * sizeof(WORD32), ALIGNMENT_16);
  /* Dummy input buffer: 3 rows */ 
  total_size += ALIGNED_SIZE(KH_3X3 * input_channels * sizeof(WORD8), ALIGNMENT_16);
  /* Rearranged kernel 3x3  */ 
  total_size += ALIGNED_SIZE(KH_3X3 * KW_3X3 * input_channels * sizeof(WORD8), ALIGNMENT_16);
  /* Rearragned NCHW kernel */
  total_size += ALIGNED_SIZE((KH_3X3 + 1) * (KW_3X3 + 1) * input_channels * sizeof(WORD8), ALIGNMENT_16);

  return total_size;
}
#endif /* DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE */

static VOID xa_nn_conv2d_depthwise_nhwc_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 y_stride
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 circ_buf_bytewidth
 )

{
    pWORD8 p_mem = p_scratch;
    xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_mem;
    int state_size;
#ifndef AE_MULAZB3X3O8X8
    state_size = ALIGNED_SIZE(sizeof(xa_nn_circ_buf_t), ALIGNMENT);
#else
    state_size = ALIGNED_SIZE(sizeof(xa_nn_circ_buf_t), ALIGNMENT_16);
#endif
    p_mem = (p_mem + state_size);
    xa_nn_circ_buf_nhwc_init(p_state
            ,p_mem
            ,circ_buf_bytewidth
            ,input_height
            ,input_channels
            ,kernel_height
            ,kernel_width
            ,channels_multiplier
            ,y_stride
            ,y_padding
            ,output_height
            );
}


static WORD32 xa_nn_conv2d_depthwise_getsize_generic
(WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 )
{
  WORD32 scratch_bytewidth = 0;
  WORD32 circ_buf_bytewidth = 0;
  WORD32 total_size = 0;

  switch (circ_buf_precision)
  {
    case 8: /* For 8b */
    case 16: /* For 16b */
      scratch_bytewidth = 8; /* 64b scratch */
      circ_buf_bytewidth = (circ_buf_precision/8); /* bytewidth as per precision */
      break;

    case -8: /* For sym16s */
      scratch_bytewidth = 8; /* 64b scratch */
      circ_buf_bytewidth = 2; /* bytewidth for sym16s */
      break;
            
    case -1: /* For f32 */
      scratch_bytewidth = 4; /* f32 scratch */
      circ_buf_bytewidth = 4; /* bytewidth for f32 */
      break;

    case -3: /* For asym8 */
    case -4: /* For asym8s */
      scratch_bytewidth = 4;
      circ_buf_bytewidth = 1;
      break;

    default:
      return -1; /* Retunrning due to invalid input */
      break;
  }

  if(inp_data_format == 0)
  {
    total_size = xa_nn_conv2d_depthwise_nhwc_getsize(input_height
        ,input_channels
        ,kernel_height
        ,kernel_width
        ,channels_multiplier
        ,y_stride
        ,y_padding
        ,output_height
        ,circ_buf_bytewidth);
  }
  else if(inp_data_format == 1)
  {
    total_size = xa_nn_conv2d_depthwise_nchw_getsize(input_width
        ,kernel_height
        ,kernel_width
        ,x_stride
        ,y_stride
        ,x_padding
        ,output_width
        ,circ_buf_bytewidth
        ,scratch_bytewidth);
  }
  return total_size;
}

WORD32 xa_nn_conv2d_depthwise_getsize
(WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 )
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_CHK_COND((x_stride <= 0), -1); //TODO: x_stride > kernel_width is supported ?
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((x_padding < 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((output_height <= 0), -1);
  XA_NNLIB_CHK_COND((output_width <= 0), -1);
  XA_NNLIB_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);

  size_t total_size_special_case = 0;
  size_t total_size_generic_case = 0;

  /* For single input channel case, use the standard convolution kernel */
  if((input_channels == 1) &&
     (circ_buf_precision == PREC_ASYM8S) &&
     (inp_data_format == 0)
     )
  {
    WORD32 out_channels = 0;/*Dummy variable for conv2d_std out_channels argument*/
    /* Alignment */
    total_size_generic_case = ALIGNMENT_16;
    /* Scratch buffer for rearranged kernel in NCHW format */ 
    total_size_generic_case += ALIGNED_SIZE(channels_multiplier * kernel_height * kernel_width, ALIGNMENT_16);

    total_size_generic_case += xa_nn_conv2d_std_getsize
       (input_height
       ,input_channels
       ,kernel_height
       ,kernel_width
       ,y_stride
       ,y_padding
       ,output_height
       ,out_channels
       ,circ_buf_precision
      );
  }
  else
  {
#ifndef DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE
#if 0 // can't check alignment of p_kernel
    if((channels_multiplier == 1) &&
        (kernel_height == 3) &&
        (kernel_width == 3) &&
        ((y_stride == 1) || (y_stride == 2)) &&
        (inp_data_format == 0) &&
        (circ_buf_precision == PREC_ASYM8S) &&
        ((input_channels & 0x3) == 0) &&
        1)
#else
    if(circ_buf_precision == PREC_ASYM8S)
#endif
    {
      total_size_special_case = xa_nn_conv2d_depthwise_getsize_k3x3
        (input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
        );
    }
#endif /* DISABLE_DEPTHWISE_CONV2D_K3X3_SPECIAL_CASE */
    {
      total_size_generic_case = xa_nn_conv2d_depthwise_getsize_generic
        (input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,output_height
         ,output_width
         ,circ_buf_precision
         ,inp_data_format
        );
    }
  }

  if(total_size_special_case > total_size_generic_case)
    return total_size_special_case;
  else
    return total_size_generic_case;
}

VOID xa_nn_conv2d_depthwise_init
(pVOID p_scratch
 ,WORD32 input_height
 ,WORD32 input_width
 ,WORD32 input_channels
 ,WORD32 kernel_height
 ,WORD32 kernel_width
 ,WORD32 channels_multiplier
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 x_padding
 ,WORD32 y_padding
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 circ_buf_precision
 ,WORD32 inp_data_format
 ,pVOID p_pad_val
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
        case -4: /* For asym8s */
            circ_buf_bytewidth = 1;

        default:
            break;
    }

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_nhwc_init(p_scratch
                ,input_height
                ,input_channels
                ,kernel_height
                ,kernel_width
                ,channels_multiplier
                ,y_stride
                ,y_padding
                ,output_height
                ,circ_buf_bytewidth);
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_nchw_init(p_scratch
                ,input_width
                ,kernel_height
                ,kernel_width
                ,x_stride
                ,y_stride
                ,x_padding
                ,output_width
                ,circ_buf_bytewidth
                ,p_pad_val);
    }
}
