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
#include <string.h>
#include "common.h"
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_err_chk.h"

WORD32 xa_nn_conv2d_std_getsize(
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 input_precision)
{
  XA_NNLIB_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
  XA_NNLIB_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_CHK_COND((y_stride <= 0), -1);
  XA_NNLIB_CHK_COND((y_padding < 0), -1);
  XA_NNLIB_CHK_COND((out_height <= 0), -1);

  WORD32 mem_req = 0;
  WORD32 input_size;
  WORD32 align_size;

  mem_req += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT);
  /* Input precision is checked here */
  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      return -1;
      break;
  }

  // Computing circular buffer size 
  // Determine y-bottom padding
  WORD32 y_b_pad = out_height * y_stride - (y_padding + input_height - kernel_height + 1);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, align_size);
  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * input_channels_pad * input_size;
  while(cir_buf_size_bytes%16 !=0)
  {
      cir_buf_size_bytes+= kernel_width*input_channels_pad*input_size;
  }
  /* scratch memory for convolution using matrix multiplication */
  mem_req += cir_buf_size_bytes;
  mem_req += BUS_WIDTH;
  
  return mem_req;
}

VOID xa_nn_conv2d_std_init_state(
    VOID *p_scratch,
    VOID *p_kernel,
    WORD32 input_height,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 input_precision)
{
  WORD8 *p_mem = (WORD8 *)p_scratch;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_mem;
  size_t input_size = 0;
  UWORD32 align_size = 0;

  switch(input_precision)
  {
    case 8:
      input_size = sizeof(WORD8);
      align_size = ALIGNMENT>>1;
      break;
    case 16:
      input_size = sizeof(WORD16);
      align_size = ALIGNMENT>>1;
      break;
    case -1:
      input_size = sizeof(WORD32);
      align_size = ALIGNMENT>>2;
      break;
    case -3:
      input_size = sizeof(UWORD8);
      align_size = ALIGNMENT>>1;
      break;
    default:
      break;
  }

  p_mem += sizeof(xa_nn_conv_state_t);
  p_mem = ALIGNED_ADDR(p_mem, ALIGNMENT);


  if(((UWORD32)p_kernel & BUS_WIDTH_MASK) == ((UWORD32)p_mem & BUS_WIDTH_MASK))
  {
    p_mem += BUS_WIDTH; /* Add a offset to avoid banking stall */
  }

  p_state->cir_buf.p_begin = p_mem;
  p_state->cir_buf.p_curr = p_mem;

  // Computing circular buffer size 
  // Determine y-bottom padding
  WORD32 y_b_pad = out_height * y_stride - (y_padding + input_height - kernel_height + 1);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  WORD32 input_channels_pad = PADDED_SIZE(input_channels, align_size);
  WORD32 cir_buf_size_bytes = (y_padding + input_height + y_b_pad) * kernel_width * input_channels_pad * input_size;
  
  while(cir_buf_size_bytes%16 !=0)
  {
      cir_buf_size_bytes+= kernel_width*input_channels_pad*input_size;
  }

  p_mem += cir_buf_size_bytes;
  p_state->cir_buf.p_end = p_mem;

  AE_SETCBEGIN0(p_state->cir_buf.p_begin);
  AE_SETCEND0(p_state->cir_buf.p_end);
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, -x_stride * input_channels_pad * input_size);

}

VOID conv2d_std_init_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);

  // Initialize circular buffer 
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<(kernel_width - x_stride);k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  if((kernel_width - x_stride) <= x_padding)
  {
    copy_x_pad_width = kernel_width - x_stride;
  }
  else 
  {
    copy_inp_width = kernel_width - x_stride - x_padding;
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  
  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<(kernel_width - x_stride);k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride x (input_height x input_channels) new planes to circular buffer
VOID conv2d_std_update_cir_buf(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;

  // Copy 'x_stride' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, x_stride * input_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width - x_stride) * input_channels_pad * input_bytewidth);

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<x_stride;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + x_stride;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = x_stride;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = x_stride;
  }

  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], 0, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<copy_x_r_pad_width;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<x_stride;k++)
    {
      memset(p_dst, 0, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
  }
  *pp_inp = (VOID *)p_inp;
}

VOID conv2d_std_init_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);

  // Initialize circular buffer 
  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<(kernel_width - x_stride);k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero and/or input data
  WORD32 copy_x_pad_width = x_padding;
  WORD32 copy_inp_width = 0;
  if((kernel_width - x_stride) <= x_padding)
  {
    copy_x_pad_width = kernel_width - x_stride;
  }
  else 
  {
    copy_inp_width = kernel_width - x_stride - x_padding;
  }
  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], pad_val_u8, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  
  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<(kernel_width - x_stride);k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, x_stride * input_channels_pad * input_bytewidth);
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;
  *pp_inp = (VOID *)p_inp;
}

// Add x_stride x (input_height x input_channels) new planes to circular buffer
VOID conv2d_std_update_cir_buf_asym8(
    WORD32 input_channels,
    WORD32 input_channels_pad,
    WORD32 input_bytewidth,
    WORD32 input_width,
    WORD32 input_height,
    WORD32 y_padding,
    WORD32 y_b_pad,
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    VOID **pp_inp,
    WORD32 idx_beg_inp_width_pad,
    xa_nn_conv_state_t *p_state,
    WORD32 pad_val)
{
  WORD32 i,k;
  WORD8 *p_inp = (WORD8 *)*pp_inp;
  UWORD8 pad_val_u8 = (UWORD8)pad_val;

  // Copy 'x_stride' planes of data to circular buffer
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_state->cir_buf.p_curr, x_stride * input_channels_pad * input_bytewidth);
  WORD8 *p_dst = (WORD8 *)p_state->cir_buf.p_curr;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width - x_stride) * input_channels_pad * input_bytewidth);

  // Set first 'y_padding' rows of cir_buf to zero
  for(i=0;i<y_padding;i++)
  {
    for(k=0;k<x_stride;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
  }

  // Set next 'input_height' rows of cir_buf with zero (from x_padding) and/or input data and/or zero (from x-right padding)
  WORD32 idx_end_inp_width_pad = idx_beg_inp_width_pad + x_stride;
  WORD32 copy_x_pad_width = 0;
  WORD32 copy_inp_width = 0;
  WORD32 copy_x_r_pad_width = 0;
  if(idx_beg_inp_width_pad < x_padding)
  {
    copy_x_pad_width = x_padding - idx_beg_inp_width_pad;
    copy_inp_width = idx_end_inp_width_pad - x_padding;
  }
  else if(idx_end_inp_width_pad <= x_padding + input_width)
  {
    copy_inp_width = x_stride;
  }
  else if(idx_beg_inp_width_pad < x_padding + input_width)
  {
    copy_inp_width = x_padding + input_width - idx_beg_inp_width_pad;
    copy_x_r_pad_width = idx_end_inp_width_pad - (x_padding + input_width);
  }
  else
  {
    copy_x_r_pad_width = x_stride;
  }

  for(i=0;i<input_height;i++)
  {
    for(k=0;k<copy_x_pad_width;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    for(k=0;k<copy_inp_width;k++)
    {
      memcpy(p_dst, p_inp, input_channels * input_bytewidth);
      memset(&p_dst[input_channels * input_bytewidth], pad_val_u8, (input_channels_pad - input_channels) * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
      p_inp += input_channels * input_bytewidth;
    }
    for(k=0;k<copy_x_r_pad_width;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
    p_inp += (input_width - copy_inp_width) * input_channels * input_bytewidth;
  }
  p_inp += (-input_height * input_width + copy_inp_width) * input_channels * input_bytewidth;

  // Set last 'y_b_pad' rows of cir_buf to zero
  for(i=0;i<y_b_pad;i++)
  {
    for(k=0;k<x_stride;k++)
    {
      memset(p_dst, pad_val_u8, input_channels_pad * input_bytewidth);
      AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, input_channels_pad * input_bytewidth);
    }
    AE_ADDCIRC16X4_XC((ae_int16x4 *)p_dst, (kernel_width-x_stride) * input_channels_pad * input_bytewidth);
  }
  *pp_inp = (VOID *)p_inp;
}
