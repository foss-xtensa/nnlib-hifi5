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
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"


#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (xthalfx4 *)((xthalf *)(ptr) + (n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }\
        else\
        {\
            ptr = (xthalfx4 *)p_scratch_zeros;\
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (xthalfx4 *)((xthalf *)(ptr) + (n) * (row_size));

#define INCR_ROW_IF_WIDTH(ptr, width, row_size) \
        if(width) \
        { \
            INCR_N_ROW(ptr, 1, row_size); \
            width--; \
        }\
        else\
        {\
            ptr = (xthalfx4 *)p_scratch_zeros;\
        }


#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(void, xa_nn_avgpool_f16_hwc,(
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
      pVOID    p_scratch_in,
      WORD16   *p_zeros_mem,
      WORD16   *p_den))
#else /* #if !HAVE_HP_VFPU */

/* Avg pooling without using extra copy of input data
 * Works with unaligned input, output.
 */
void xa_nn_avgpool_f16_hwc(
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
      pVOID    p_scratch_in,
      WORD16   *p_zeros_mem,
      WORD16   *p_den)
{
    xthalf *p_scratch = (xthalf *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    xthalfx4 * p_src1, * p_src2, * p_src3;
    xthalfx4 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    xthalfx4 *p_dst, *p_dst_temp;
    ae_valign align_src1, align_src2, align_src3, align_dst;
    int i;
    xthalf *p_dst_pad, *p_rec_den;

    plane_size = input_width * input_channels;

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;
        xthalf *p_scratch_zeros = (xthalf *)p_zeros_mem;


        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating sum of k_h w-c planes and saving into the scratch memory*/
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (xthalfx4 *)p_scratch;


        if(pool_height)
        {
            p_src1 = (xthalfx4 *)p_inp;
            INCR_N_PLANE(p_src1, start_plane, plane_size);
            pool_height--;

            p_src2 = p_src1;
            INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

            p_src3 = p_src2;
            INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            /* Add three rows per iteration */
            do
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                /* prime */
                align_src1 = AE_LA64_PP(p_src1_temp);
                align_src2 = AE_LA64_PP(p_src2_temp);
                align_src3 = AE_LA64_PP(p_src3_temp);
                align_dst = AE_ZALIGN64();

                for(i = 0; i < (plane_size >> 2); i++)
                {
                    xthalfx4 temp, i1, i2, i3, out;

                    AE_LAHX4IP(i1, align_src1, p_src1_temp);
                    AE_LAHX4IP(i2, align_src2, p_src2_temp);
                    AE_LAHX4IP(i3, align_src3, p_src3_temp);

                    temp = ADD_HX4(i1, i2);
                    out  = ADD_HX4(temp, i3);

                    AE_SAHX4IP(out, align_dst, p_dst_temp);
                }

                AE_SA64POS_FP(align_dst, p_dst_temp);

                /* remainder loop */
                if(plane_size & 3)
                {
                    xthalf temp, i1h, i2h, i3h, outh;
                    int rem_idx;

                    for(rem_idx = 0; rem_idx < (plane_size & 3); rem_idx++)
                    {
                        i1h = ((xthalf *)p_src1_temp)[rem_idx];
                        i2h = ((xthalf *)p_src2_temp)[rem_idx];
                        i3h = ((xthalf *)p_src3_temp)[rem_idx];

                        temp = ADD_H(i1h, i2h);
                        outh = ADD_H(temp, i3h);
                        ((xthalf *)p_dst_temp)[rem_idx] = outh;
                    }
                }

                if(!pool_height)
                    break;

                p_src1 = p_dst;

                p_src2 = p_src3;
                INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                p_src3 = p_src2;
                INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            }while(1);
        }
        else
        {
            /* If there is no valid input present, fill the output with zeros*/
            p_dst_pad = (xthalf *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] = CONST_H(0);
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        p_rec_den = (xthalf *)p_den + itr_oh*out_width;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            xthalf recip_den_s;
            xthalf *p_scratch_zeros = (xthalf *)p_zeros_mem;

            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_dst = (xthalfx4 *)((xthalf *)p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels));
            recip_den_s = p_rec_den[0];
            p_rec_den++;

            if(pool_width)
            {
                p_src1 = (xthalfx4 *)p_scratch;
                INCR_N_ROW(p_src1, start_row, input_channels);
                pool_width--;

                p_src2 = p_src1;
                INCR_ROW_IF_WIDTH(p_src2, pool_width, input_channels);

                p_src3 = p_src2;
                INCR_ROW_IF_WIDTH(p_src3, pool_width, input_channels);

                /* Add three rows per iteration */
                do
                {
                    p_dst_temp = p_dst;
                    p_src1_temp = p_src1;
                    p_src2_temp = p_src2;
                    p_src3_temp = p_src3;

                    /* prime */
                    align_src1 = AE_LA64_PP(p_src1_temp);
                    align_src2 = AE_LA64_PP(p_src2_temp);
                    align_src3 = AE_LA64_PP(p_src3_temp);

                    align_dst = AE_ZALIGN64();

                    for(i = 0; i < (input_channels >> 2); i++)
                    {
                        xthalfx4 i1, i2, i3, out;

                        AE_LAHX4IP(i1, align_src1, p_src1_temp);
                        AE_LAHX4IP(i2, align_src2, p_src2_temp);
                        AE_LAHX4IP(i3, align_src3, p_src3_temp);

                        out = ADD_HX4(i1, i2);
                        out = ADD_HX4(out, i3);

                        AE_SAHX4IP(out, align_dst, p_dst_temp);
                    }

                    AE_SA64POS_FP(align_dst, p_dst_temp);

                    /* remainder loop */
                    if(input_channels & 3)
                    {
                        xthalf i1h, i2h, i3h, outh;
                        int rem_idx;

                        for(rem_idx = 0; rem_idx < (input_channels & 3); rem_idx++)
                        {
                            i1h = ((xthalf *)p_src1_temp)[rem_idx];
                            i2h = ((xthalf *)p_src2_temp)[rem_idx];
                            i3h = ((xthalf *)p_src3_temp)[rem_idx];

                            outh = ADD_H(i1h, i2h);
                            outh = ADD_H(outh, i3h);

                            ((xthalf *)p_dst_temp)[rem_idx] = outh;
                        }
                    }


                    if(!pool_width)
                        break;

                    p_src1 = p_dst;

                    p_src2 = p_src3;
                    INCR_ROW_IF_WIDTH(p_src2, pool_width, input_channels);

                    p_src3 = p_src2;
                    INCR_ROW_IF_WIDTH(p_src3, pool_width, input_channels);

                }while(1);

                /* Multiply by reciprocal of denominator */
                p_dst_pad = (xthalf *)p_dst;
                for(i = 0; i < input_channels; i++)
                {
                    xthalf i1, out;

                    i1 = p_dst_pad[i];
                    out = MUL_H(i1, recip_den_s);
                    p_dst_pad[i] = out;
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros */
                p_dst_pad = (xthalf *)p_dst;
                for(i = 0; i < input_channels; i++)
                {
                    p_dst_pad[i] = CONST_H(0);
                }
            }
        }
    }
}

#endif /* #if !HAVE_HP_VFPU */
