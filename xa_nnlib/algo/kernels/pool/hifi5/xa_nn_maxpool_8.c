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
#include "xa_nn_maxpool_state.h"
#include "xa_nnlib_err_chk.h"

#define INCR_N_ROW(ptr, n) \
    ptr = (ae_int8x8 *)((WORD8 *)(ptr) + (n) * (input_width));

#define INCR_ROW_IF_HEIGHT(ptr, height) \
        if(height) \
        { \
            INCR_N_ROW(ptr, 1); \
            height--; \
        }

#define INC_1_IF_WIDTH(ptr, width) \
        if(width) \
        { \
            ptr = (ae_int8x8 *)((WORD8 *)ptr + 1); \
            width--; \
        }

#define MAX_16X4(id1, id0) \
        b0 = AE_LT16(id1, id0); \
        AE_MOVT16X4(id1, id0, b0); \


/* Max pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

static void maxpool_8(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_inp,
    WORD32  input_height,
    WORD32   input_width,
    WORD32   kernel_height,
    WORD32   kernel_width,
    WORD32   x_stride,
    WORD32   y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32   out_height,
    WORD32   out_width,
    pVOID    p_scratch_in)
{
    WORD8 *p_scratch = (WORD8 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    ae_int8x8 * p_src1, * p_src2, * p_src3;
    ae_int8x8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int8x8 *p_dst, *p_dst_temp;
    ae_valignx2 align_s1, align_s2, align_s3;
    int i;
    WORD8 *p_dst_pad;

    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD8));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = 0x80;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = 0x80;
    }

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;

        /* Pool height processing */

        /* Compare the input rows for the required pooling height and store on scratch */
        start_row  = itr_oh * y_stride - y_padding;
        end_row = start_row + kernel_height;
        LIMIT(start_row , 0, input_height);
        LIMIT(end_row , 0, input_height);

        pool_height = end_row - start_row;

        p_dst = (ae_int8x8 *)((WORD8 *)p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (ae_int8x8 *)p_inp;
            INCR_N_ROW(p_src1, start_row);
            pool_height--;

            p_src2 = p_src1;
            INCR_ROW_IF_HEIGHT(p_src2, pool_height);

            p_src3 = p_src2;
            INCR_ROW_IF_HEIGHT(p_src3, pool_height);

            /* Compare three rows per iteration */
            do
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                /* prime */
                align_s1 = AE_LA128_PP(p_src1_temp);
                align_s2 = AE_LA128_PP(p_src2_temp);
                align_s3 = AE_LA128_PP(p_src3_temp);

                //TODO: improve packing, use AE_MAX32 ?
                for(i = 0; i < (input_width >> 4); i++)
                {
                    ae_int8x8 i1, i2, i3, temp;
                    ae_int8x8 j1, j2, j3;

                    AE_LA8X8X2_IP(i1, j1, align_s1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i2, j2, align_s2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i3, j3, align_s3, (ae_int8x16 *)p_src3_temp);

                    temp = AE_MAX8(i2, i3);
                    i2 = AE_MAX8(temp, i1);
                    temp = AE_MAX8(j2, j3);
                    j2 = AE_MAX8(temp, j1);

                    AE_S8X8X2_IP(i2, j2, (ae_int8x16 *)p_dst_temp, 16);
                }

                /* remainder loop for input_width */
                for(i = 0; i < (input_width & 15); i++)
                {
                    ae_int8x8 i1, i2, i3, temp;

                    AE_L8_IP(i1, (ae_int8 *)p_src1_temp, 1);
                    AE_L8_IP(i2, (ae_int8 *)p_src2_temp, 1);
                    AE_L8_IP(i3, (ae_int8 *)p_src3_temp, 1);

                    temp = AE_MAX8(i2, i3);
                    i2 = AE_MAX8(temp, i1);

                    AE_S8_0_IP(i2, (ae_int8 *)p_dst_temp, 1);
                }


                if(!pool_height)
                    break;

                p_src1 = p_dst;

                p_src2 = p_src3;
                INCR_ROW_IF_HEIGHT(p_src2, pool_height);

                p_src3 = p_src2;
                INCR_ROW_IF_HEIGHT(p_src3, pool_height);

            }while(1);
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst_pad[i] = 0x80;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD8));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (ae_int8x8 *)((WORD8 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_src1 = (ae_int8x8 *)((WORD8 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;

        p_src2 = p_src1;
        INC_1_IF_WIDTH(p_src2, pool_width);

        p_src3 = p_src2;
        INC_1_IF_WIDTH(p_src3, pool_width);

        do
        {
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;
            p_src2_temp = p_src2;
            p_src3_temp = p_src3;

            /* prime */
            align_s1 = AE_LA128_PP(p_src1_temp);
            align_s2 = AE_LA128_PP(p_src2_temp);
            align_s3 = AE_LA128_PP(p_src3_temp);

            for(i = 0; i < (scratch_width >> 4); i++)
            {
                ae_int8x8 src1, src2, src3, temp;
                ae_int8x8 j1, j2, j3;

                AE_LA8X8X2_IP(src1, j1, align_s1, (ae_int8x16 *)p_src1_temp);
                AE_LA8X8X2_IP(src2, j2, align_s2, (ae_int8x16 *)p_src2_temp);
                AE_LA8X8X2_IP(src3, j3, align_s3, (ae_int8x16 *)p_src3_temp);

                temp = AE_MAX8(src2, src3);
                src2 = AE_MAX8(temp, src1);
                temp = AE_MAX8(j2, j3);
                j2 = AE_MAX8(temp, j1);

                AE_S8X8X2_IP(src2, j2, (ae_int8x16 *)p_dst_temp, 16);
            }

            /* remainder loop for scratch_width */
            for(i = 0; i < (scratch_width & 15); i++)
            {
                ae_int8x8 src1, src2, src3, temp;

                AE_L8_IP(src1, (ae_int8 *)p_src1_temp, 1);
                AE_L8_IP(src2, (ae_int8 *)p_src2_temp, 1);
                AE_L8_IP(src3, (ae_int8 *)p_src3_temp, 1);

                temp = AE_MAX8(src2, src3);
                src2 = AE_MAX8(temp, src1);

                AE_S8_0_IP(src2, (ae_int8 *)p_dst_temp, 1);
            }

            if(!pool_width)
                break;

            /* Setup next iteration */
            p_src1 = p_dst;
            p_src2 = p_src3;
            INC_1_IF_WIDTH(p_src2, pool_width);
            p_src3 = p_src2;
            INC_1_IF_WIDTH(p_src3, pool_width);

        }while(1);

        WORD8 *ptr_out1 = p_scratch + total_out_width;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh * out_width + itr_ow] =  ptr_out1[itr_ow * x_stride];
        }
    }
}

WORD32 xa_nn_maxpool_8(
    WORD8* __restrict__ p_out,
    WORD8* __restrict__ p_inp,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    VOID   *p_scratch)
{
    WORD32 err = 0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
    
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);
    // Different I/O data formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);

    if((input_channels == 1) || (out_data_format == 1))
    {
        err = xa_nn_maxpool_init(8
                                 ,p_scratch
                                 ,input_width
                                 ,kernel_height
                                 ,kernel_width
                                 ,x_stride
                                 ,y_stride
                                 ,x_padding
                                 ,out_width
                                 );
        if(err<0)
            return err;

        xa_nn_maxpool_state_t *p_state = (xa_nn_maxpool_state_t *)p_scratch;
        WORD8 *p_scratch_in = (WORD8 *)(p_state->p_scratch);
        int itr_ic;
        WORD8 *pt_inp, *pt_out;

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            maxpool_8(pt_out
                    ,pt_inp
                    ,input_height
                    ,input_width
                    ,kernel_height
                    ,kernel_width
                    ,x_stride
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,out_height
                    ,out_width
                    ,p_scratch_in
                    );
            }
        }
#if 1 //NHWC format support
    else
    {
        xa_nn_maxpool_8_hwc(
                p_out,
                p_inp,
                input_height,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                x_stride,
                y_stride,
                x_padding,
                y_padding,
                out_height,
                out_width,
                p_scratch);

    }
#endif
    return 0;
}
