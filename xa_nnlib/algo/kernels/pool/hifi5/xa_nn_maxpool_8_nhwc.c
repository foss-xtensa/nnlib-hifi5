/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_maxpool_state.h"
#include <math.h>
#include <string.h>

#define INCR_N_PLANE_1(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (ptr) + ((n) * (row_size));

#define INCR_ROW_IF_WIDTH(ptr, width, row_size) \
        if(width) \
        { \
            INCR_N_ROW(ptr, 1, row_size); \
            width--; \
        }

#define MAX_16X4(out, id2, id1, id0) {\
        out = id1;\
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(out, id0, b0);\
        b0 = AE_LT16(out, id2); \
        AE_MOVT16X4(out, id2, b0);\
}

/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

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
      pVOID    p_scratch_in)
{
    WORD8 *p_scratch = (WORD8 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD8 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int8x8 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int8x8 * p_dst, *p_dst_temp;
    WORD8 *p_out_temp;
    ae_int8x8 * p_src1_scratch;
    ae_valignx2 align_src1, align_src2, align_src3, align_dst;
    int i;
    WORD8 *p_dst_pad;
    
    int left_pad_aligned, right_pad, total_out_width;
    
    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT);

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    memset(p_dst_pad, (WORD8)0x80, left_pad_aligned*input_channels);
    
    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + (left_pad_aligned + input_width)*input_channels;
    memset(p_dst_pad, (WORD8)0x80, (right_pad + kernel_width)*input_channels);

    plane_size = input_width * input_channels;
    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row;
        int start_plane, end_plane;

        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating max of k_h w-c planes and saving into the scratch memory*/
        /* Compare the input w-c planes (width-channel planes) for the required pooling height and store to the scratch */
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int8x8 *)((WORD8 *)p_scratch + (left_pad_aligned*input_channels));

        if(pool_height)
        {
            p_src1 = (WORD8 *)p_inp;
            INCR_N_PLANE(p_src1, start_plane, plane_size);
            pool_height--;

            p_src2 = p_src1;
            INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

            p_src3 = p_src2;
            INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            align_dst = AE_ZALIGN128(); // zero alignment reg
            /* 1st instance: Compare three rows per iteration */
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                align_src1 = AE_LA128_PP(p_src1_temp);
                align_src2 = AE_LA128_PP(p_src2_temp);
                align_src3 = AE_LA128_PP(p_src3_temp);

                for(i = 0; i < (plane_size >> 4); i++)
                {
                    ae_int8x8 i1, i2, i3, j1, j2, j3;
                    ae_int8x8 out, out1;
                    AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                    out = AE_MAX8(i1, i2);
                    out = AE_MAX8(out, i3);
                    out1 = AE_MAX8(j1, j2);
                    out1 = AE_MAX8(out1, j3);
                    AE_SA8X8X2_IP(out, out1, align_dst, (ae_int8x16 *)p_dst_temp);
                }

                AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop */
                for(i = 0; i < (plane_size & 15); i++)
                {
                    ae_int8x8 i1, i2, i3;
                    ae_int8x8 out;
                    AE_L8_IP(i1, (ae_int8 *)p_src1_temp,1);
                    AE_L8_IP(i2, (ae_int8 *)p_src2_temp,1);
                    AE_L8_IP(i3, (ae_int8 *)p_src3_temp,1);

                    out = AE_MAX8(i1, i2);
                    out = AE_MAX8(out, i3);
                    AE_S8_0_IP(out, (ae_int8 *)p_dst_temp, 1);
                }
            }

            if(pool_height)
            {
                p_src2 = p_src3;
                INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                p_src3 = p_src2;
                INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                do
                {
                    p_dst_temp = p_dst;
                    p_src1_scratch = p_dst;
                    p_src2_temp = p_src2;
                    p_src3_temp = p_src3;

                    align_src2 = AE_LA128_PP(p_src2_temp);
                    align_src3 = AE_LA128_PP(p_src3_temp);

                    align_dst = AE_ZALIGN128(); // zero alignment reg
                    align_src1 = AE_LA128_PP(p_src1_scratch);

                    for(i = 0; i < (plane_size >> 4); i++)
                    {
                        ae_int8x8 i1, i2, i3, j1, j2, j3;
                        ae_int8x8 out, out1;
                        AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_scratch);
                        AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                        AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                        out = AE_MAX8(i1, i2);
                        out = AE_MAX8(out, i3);
                        out1 = AE_MAX8(j1, j2);
                        out1 = AE_MAX8(out1, j3);
                        AE_SA8X8X2_IP(out, out1, align_dst, (ae_int8x16 *)p_dst_temp);
                    }

                    AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 15); i++)
                    {
                        ae_int8x8 i1, i2, i3;
                        ae_int8x8 out;
                        AE_L8_IP(i1, (ae_int8 *)p_src1_scratch,1);
                        AE_L8_IP(i2, (ae_int8 *)p_src2_temp,1);
                        AE_L8_IP(i3, (ae_int8 *)p_src3_temp,1);

                        out = AE_MAX8(i1, i2);
                        out = AE_MAX8(out, i3);
                        AE_S8_0_IP(out, (ae_int8 *)p_dst_temp, 1);
                    }

                    if(!pool_height)
                        break;

                    p_src2 = p_src3;
                    INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                    p_src3 = p_src2;
                    INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                }while(1);
            }
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = ((WORD8 *)p_scratch + (left_pad_aligned*input_channels));
            memset(p_dst_pad, (WORD8)0x80, plane_size);
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        if(input_channels < 16)
        {
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              start_row  = itr_ow * x_stride + left_pad_aligned - x_padding;
              pool_width = kernel_width;
              p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow * input_channels);

              int rem_inp_chan = input_channels & 0xf;

              p_src1_w = (WORD8 *)p_scratch;
              INCR_N_ROW(p_src1_w, start_row, input_channels);
              pool_width--;

              p_src2_w = p_src1_w;
              INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

              p_src3_w = p_src2_w;
              INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

              /* Compare three rows per iteration */
              do
              {
                  p_dst_temp = (ae_int8x8 *)p_out_temp;
                  p_src1_temp_w = (ae_int8x8 *)p_src1_w;
                  p_src2_temp_w = (ae_int8x8 *)p_src2_w;
                  p_src3_temp_w = (ae_int8x8 *)p_src3_w;

                  /* prime */
                  align_src1 = AE_LA128_PP(p_src1_temp_w);
                  align_src2 = AE_LA128_PP(p_src2_temp_w);
                  align_src3 = AE_LA128_PP(p_src3_temp_w);
                  align_dst = AE_ZALIGN128(); // zero alignment reg

                  ae_int8x8 i1, i2, i3, j1, j2, j3;
                  ae_int8x8 out, out1;
                  
                  AE_LAV8X8X2_XP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp_w, rem_inp_chan);
                  AE_LAV8X8X2_XP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp_w, rem_inp_chan);
                  AE_LAV8X8X2_XP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp_w, rem_inp_chan);

                  out = AE_MAX8(i1, i2);
                  out = AE_MAX8(out, i3);
                  out1 = AE_MAX8(j1, j2);
                  out1 = AE_MAX8(out1, j3);

                  AE_SAV8X8X2_XP(out, out1, align_dst, (ae_int8x16 *)p_dst_temp, rem_inp_chan);
                  AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                  if(!pool_width)
                      break;

                  p_src1_w = (WORD8 *)p_out_temp;

                  p_src2_w = p_src3_w;
                  INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

                  p_src3_w = p_src2_w;
                  INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

                  }while(1);
          }
        }
        else
        {
          for(itr_ow = 0; itr_ow < out_width; itr_ow++)
          {
              start_row  = itr_ow * x_stride + left_pad_aligned - x_padding;
              pool_width = kernel_width;
              p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow * input_channels);

              int rem_inp_chan = input_channels & 0xf;

                  p_src1_w = (WORD8 *)p_scratch;
                  INCR_N_ROW(p_src1_w, start_row, input_channels);
                  pool_width--;

                  p_src2_w = p_src1_w;
                  INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

                  p_src3_w = p_src2_w;
                  INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

                  /* Compare three rows per iteration */
                  do
                  {
                      p_dst_temp = (ae_int8x8 *)p_out_temp;
                      p_src1_temp_w = (ae_int8x8 *)p_src1_w;
                      p_src2_temp_w = (ae_int8x8 *)p_src2_w;
                      p_src3_temp_w = (ae_int8x8 *)p_src3_w;

                      /* prime */
                      align_src1 = AE_LA128_PP(p_src1_temp_w);
                      align_src2 = AE_LA128_PP(p_src2_temp_w);
                      align_src3 = AE_LA128_PP(p_src3_temp_w);
                      align_dst = AE_ZALIGN128(); // zero alignment reg

                      ae_int8x8 i1, i2, i3, j1, j2, j3;
                      ae_int8x8 out, out1;
                      for(i = 0; i < (input_channels >> 4); i++)
                      {
                          AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp_w);
                          AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp_w);
                          AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp_w);

                          out = AE_MAX8(i1, i2);
                          out = AE_MAX8(out, i3);
                          out1 = AE_MAX8(j1, j2);
                          out1 = AE_MAX8(out1, j3);

                          AE_SA8X8X2_IP(out, out1, align_dst, (ae_int8x16 *)p_dst_temp);
                      }
                      if(rem_inp_chan)
                      {
                          AE_LAV8X8X2_XP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp_w, rem_inp_chan);
                          AE_LAV8X8X2_XP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp_w, rem_inp_chan);
                          AE_LAV8X8X2_XP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp_w, rem_inp_chan);

                          out = AE_MAX8(i1, i2);
                          out = AE_MAX8(out, i3);
                          out1 = AE_MAX8(j1, j2);
                          out1 = AE_MAX8(out1, j3);

                          AE_SAV8X8X2_XP(out, out1, align_dst, (ae_int8x16 *)p_dst_temp, rem_inp_chan);
                      }
                      AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                      if(!pool_width)
                          break;

                      p_src1_w = (WORD8 *)p_out_temp;

                      p_src2_w = p_src3_w;
                      INCR_ROW_IF_WIDTH(p_src2_w, pool_width, input_channels);

                      p_src3_w = p_src2_w;
                      INCR_ROW_IF_WIDTH(p_src3_w, pool_width, input_channels);

                  }while(1);
          }
        }
    }
}


