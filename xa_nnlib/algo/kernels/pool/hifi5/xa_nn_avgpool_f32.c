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
#include "common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_avgpool_f32,(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp,
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
    WORD32  out_data_format,
    VOID *handle))
#else /* #if !HAVE_VFPU */
static void avgpool_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  out_plane_size,
    WORD32  not_last_channel,
    pVOID   p_scratch_in)
{
    FLOAT32 *p_scratch = (FLOAT32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    const xtfloatx2 * p_src1, * p_src2;
    const xtfloatx2 * __restrict p_src1_temp, * __restrict p_src2_temp; 
    xtfloatx2 *p_dst, *p_dst_temp;
    ae_valignx2 align_src1, align_src2;
    int i;
    FLOAT32 *p_dst_pad;


    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(FLOAT32));

    /* Left padding of temporary output with min_value */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = 0.0f;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width); 
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = 0.0f;
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

        p_dst = (xtfloatx2 *)((FLOAT32 *)p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (const xtfloatx2 *)p_inp;
            p_src1 = (const xtfloatx2 *)((const FLOAT32 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;

            /* prime */
            align_src1 = AE_LA128_PP(p_src1_temp);

            for(i = 0; i < (input_width >> 2); i++)
            {
                xtfloatx2 i1, j1;
                AE_LASX2X2_IP(i1, j1, align_src1, (const xtfloatx4 *)p_src1_temp);
                AE_SSX2X2_IP(i1, j1, (xtfloatx4 *)p_dst_temp, 16);
            }

            /* reminder loop for input_width */
            for(i=0;i<(input_width & 3);i++)
            {
                xtfloatx2 out;
                out = ((const FLOAT32 *)p_src1_temp)[i];
                ((FLOAT32 *)p_dst_temp)[i] = out;
            }

            p_src2 = p_src1;
            p_src1 = p_dst;
            /* Compare three rows per iteration */
            while(pool_height)
            {
                p_src2 = (const xtfloatx2 *)((const FLOAT32 *)p_src2 + input_width);
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;

                /* prime */
                align_src2 = AE_LA128_PP(p_src2_temp);

                for(i = 0; i < (input_width >> 2); i++)
                {
                    xtfloatx2 i1, j1, i2, j2, out, out1;

                    AE_LSX2X2_IP(i1, j1, (const xtfloatx4 *)p_src1_temp, 16);
                    AE_LASX2X2_IP(i2, j2, align_src2, (const xtfloatx4 *)p_src2_temp);

                    ADD_SX2X2(out, out1, i1, j1, i2, j2);
                    AE_SSX2X2_IP(out, out1, (xtfloatx4 *)p_dst_temp, 16);
                }

                /* reminder loop for input_width */
                for(i=0;i<(input_width & 3);i++)
                {
                    xtfloatx2 i1, i2, out;

                    i1 = ((const FLOAT32 *)p_src1_temp)[i];
                    i2 = ((const FLOAT32 *)p_src2_temp)[i];

                    out = ADD_SX2(i1, i2);
                    ((FLOAT32 *)p_dst_temp)[i] = out;
                }
                pool_height--;
            };
        }
        else
        {
            /* If there is no valid input present, fill the output with min_value */
            p_dst_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst_pad[i] = 0.0f;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(FLOAT32));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (xtfloatx2 *)((FLOAT32 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_src1 = (const xtfloatx2 *)((FLOAT32 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;
        p_dst_temp = p_dst;
        p_src1_temp = p_src1;

        /* prime */
        align_src1 = AE_LA128_PP(p_src1_temp);

        for(i = 0; i < (scratch_width >> 2); i++)
        {
            xtfloatx2 src1, src2;
            AE_LASX2X2_IP(src1, src2, align_src1, (const xtfloatx4 *)p_src1_temp);
            AE_SSX2X2_IP(src1, src2, (xtfloatx4 *)p_dst_temp, 16);
        }

        /* reminder loop for scratch_width */
        for(i=0;i<(scratch_width & 3);i++)
        {
           xtfloatx2 src1;
           src1 = ((const FLOAT32 *)p_src1_temp)[i];
           ((FLOAT32 *)p_dst_temp)[i] = src1;
        }

        p_src2 = p_src1;
        p_src1 = p_dst;

        while(pool_width > 0)
        {
            p_src2 = (const xtfloatx2 *)((const FLOAT32 *)p_src2 + 1);
            p_dst_temp = p_dst;
            p_src1_temp = p_src1;
            p_src2_temp = p_src2;

            /* prime */
            align_src2 = AE_LA128_PP(p_src2_temp);

            for(i = 0; i < (scratch_width >> 2); i++)
            {
                xtfloatx2 src1, src2, src3, src4, out, out1;
                AE_LSX2X2_IP(src1, src3, (const xtfloatx4 *)p_src1_temp, 16);
                AE_LASX2X2_IP(src2, src4, align_src2, (const xtfloatx4 *)p_src2_temp);
                ADD_SX2X2(out, out1, src1, src3, src2, src4);
                AE_SSX2X2_IP(out, out1, (xtfloatx4 *)p_dst_temp, 16);
            }

            /* remainder loop for scratch_width */
             for(i=0;i<(scratch_width & 3);i++)
             {
                xtfloatx2 src1, src2, out;

                src1 = ((const FLOAT32 *)p_src1_temp)[i];
                src2 = ((const FLOAT32 *)p_src2_temp)[i];

                out = ADD_SX2(src1, src2);
                ((FLOAT32 *)p_dst_temp)[i] = out;
             }
             pool_width--;
        };

        FLOAT32 *ptr_out1 = (FLOAT32 *)((FLOAT32 *)p_scratch + total_out_width);
        FLOAT32 den_inv, den1_inv;
        if(not_last_channel)
        {
            itr_ow = 0;
            den_inv = p_out[itr_oh*out_width+itr_ow];
            den1_inv = p_out[itr_oh*out_width+itr_ow+1];
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_S(ptr_out1[itr_ow*x_stride], den_inv);
                p_out[itr_oh*out_width+itr_ow+1] = MUL_S(ptr_out1[itr_ow*x_stride+x_stride], den1_inv);
                /* store 1/den for next channel */
                p_out[out_plane_size + itr_oh*out_width+itr_ow] = den_inv;
                p_out[out_plane_size + itr_oh*out_width+itr_ow+1] = den1_inv;
                den_inv = p_out[itr_oh*out_width+itr_ow+2];
                den1_inv = p_out[itr_oh*out_width+itr_ow+3];
            }
            if(out_width & 1)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_S(ptr_out1[itr_ow*x_stride], den_inv);
                /* store 1/den for next channel */
                p_out[out_plane_size + itr_oh*out_width+itr_ow] = den_inv;

            }
        }
        else
        {
            itr_ow = 0;
            den_inv = p_out[itr_oh*out_width+itr_ow];
            den1_inv = p_out[itr_oh*out_width+itr_ow+1];
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_S(ptr_out1[itr_ow*x_stride], den_inv);
                p_out[itr_oh*out_width+itr_ow+1] = MUL_S(ptr_out1[itr_ow*x_stride+x_stride], den1_inv);
                den_inv = p_out[itr_oh*out_width+itr_ow+2];
                den1_inv = p_out[itr_oh*out_width+itr_ow+3];
            }
            if(out_width & 1)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_S(ptr_out1[itr_ow*x_stride], den_inv);

            }
        }
    }
}

WORD32 xa_nn_avgpool_f32(
    FLOAT32* __restrict__ p_out,
    const FLOAT32* __restrict__ p_inp,
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
    WORD32  out_data_format,
    VOID *p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 1), -1);

    xa_nn_avgpool_init(-1,
                       p_scratch,
                       input_width,
                       kernel_height,
                       kernel_width,
                       x_stride,
                       y_stride,
                       x_padding,
                       out_height,
                       out_width);

    xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_scratch;
    FLOAT32 *p_tmp_out = (FLOAT32 *)(p_state->p_tmp_out);
    int itr_ic, itr_oh, itr_ow;
    const FLOAT32 *pt_inp;
    FLOAT32 *pt_out;

    /* Calculate denominators for division */
    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
        kernel_y_start = itr_oh*y_stride - y_padding;
        kernel_y_end = kernel_y_start + kernel_height;
        LIMIT(kernel_y_start, 0, input_height)
        LIMIT(kernel_y_end, 0, input_height)
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            kernel_x_start = itr_ow*x_stride - x_padding;
            kernel_x_end = kernel_x_start + kernel_width;
            LIMIT(kernel_x_start, 0, input_width)
            LIMIT(kernel_x_end, 0, input_width)
            FLOAT32 den = (FLOAT32)((kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start));
            p_out[itr_oh*out_width+itr_ow] = MAX_S(RECIP_S(den), 0.0f);
        }
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic * input_height * input_width];
        pt_out = &p_out[itr_ic * out_height * out_width];

        avgpool_f32(pt_out
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
                ,out_height*out_width
                ,(input_channels-itr_ic-1)
                ,p_tmp_out
                );
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */
