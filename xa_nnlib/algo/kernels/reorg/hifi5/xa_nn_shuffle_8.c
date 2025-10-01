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
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

WORD32 xa_nn_shuffle_3D_8_8(WORD8 * __restrict__ p_out
                    ,const WORD8 * __restrict__ p_inp
                    ,WORD32 input_height
                    ,WORD32 input_width
                    ,WORD32 input_channel
                    ,WORD32 output_height
                    ,WORD32 output_width
                    ,WORD32 output_channel
                    ,WORD32 interleave_groups)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((input_height != output_height),-1);
  XA_NNLIB_ARG_CHK_COND((input_width != output_width),-1);
  XA_NNLIB_ARG_CHK_COND((input_channel != output_channel),-1);
  XA_NNLIB_ARG_CHK_COND((interleave_groups < 0 || interleave_groups > output_channel),-1);
  XA_NNLIB_ARG_CHK_COND((output_channel % interleave_groups != 0),-1);
  XA_NNLIB_ARG_CHK_COND((input_height < 0),-1);
  XA_NNLIB_ARG_CHK_COND((input_width < 0),-1);
  XA_NNLIB_ARG_CHK_COND((input_channel < 0),-1);
  
  WORD32 channel_per_group = output_channel / interleave_groups;
  WORD32 hw_plane_count = output_height*output_width;
  
  WORD8 *inp_ptr = (WORD8 *)p_inp;
  ae_valign inp_align = AE_LA64_PP(inp_ptr);
  ae_int8x8 d_inp;
  ae_int8 *out_ptr;
  WORD32 w,x,y;
    
  if(interleave_groups==1 || interleave_groups==output_channel)
  {
    WORD8* ptr_out = p_out;
    MEMCPY_8b(ptr_out, inp_ptr, hw_plane_count*output_channel);
  }
  else if(channel_per_group < 32 && interleave_groups > channel_per_group)
  {
    out_ptr = (ae_int8 *)p_out;
    for(y=0; y<hw_plane_count; y++)
    {
      for(w=0; w<channel_per_group; w++)
      {
        ae_int8 *inp_ptr2 = (ae_int8 *)(p_inp + y * output_channel + w);
        for(x=0; x<interleave_groups; x++)
        {
          AE_L8_XP(d_inp, inp_ptr2, channel_per_group);
          AE_S8_0_XP(d_inp, out_ptr, 1);
        }
      }
    }
  }
  else
  {
    for(x=0; x<interleave_groups; x++)
    {
      out_ptr = (ae_int8 *)(p_out + x);
      for(y=0; y<hw_plane_count; y++)
      {
        ae_int8x8 *inp8x8_ptr = (ae_int8x8 *)(p_inp + x * channel_per_group + y * output_channel);
        inp_align = AE_LA64_PP(inp8x8_ptr);
        for(w=0; w<(channel_per_group & ~(8-1)); w+=8)
        {
          AE_LA8X8_IP(d_inp,inp_align,inp8x8_ptr);
          AE_SW_S8_7_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_6_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_5_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_4_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_3_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_2_XP(d_inp, out_ptr, interleave_groups);
          AE_SW_S8_1_XP(d_inp, out_ptr, interleave_groups);
          AE_S8_0_XP(d_inp, out_ptr, interleave_groups);
        }
        ae_int8 *inp8_ptr = (ae_int8 *)inp8x8_ptr;
        for(;w<channel_per_group;w++)
        {
          AE_L8_IP(d_inp, inp8_ptr, 1);
          AE_S8_0_XP(d_inp, out_ptr, interleave_groups);
        }
        inp8x8_ptr = (ae_int8x8 *)inp8_ptr;
      }
    }
  }
  return 0;
}