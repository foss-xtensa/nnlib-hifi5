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

#if HAVE_HP_VFPU
static void avgpool_f16(
    WORD16* __restrict__ pt_out,
    const WORD16* __restrict__ p_inp,
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
    xthalf *p_scratch = (xthalf *)(p_scratch_in);

    xthalf* __restrict__ p_out = (xthalf*)pt_out;
    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    const xthalfx4 * p_src1, * p_src2;
    const xthalfx4 * __restrict__ p_src1_temp, * __restrict__ p_src2_temp;
    xthalfx4 *p_dst;
    xthalfx4 *p_dst_temp;
    ae_valign align_src1, align_src2;
    int i;
    xthalf *p_dst_pad;

    /* Match scratch size alignment used by getsize (align to 8 halfs) */
    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD16));

    /* Left padding of temporary output with zero */
    p_dst_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst_pad[i] = CONST_H(0);
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with zero,
     * add kernel_width values more for the aligning load operations */
    p_dst_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst_pad[i] = CONST_H(0);
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

        p_dst = (xthalfx4 *)(p_scratch + left_pad_aligned);

        if(pool_height)
        {
            p_src1 = (const xthalfx4 *)p_inp;
            p_src1 = (const xthalfx4 *)((const xthalf *)p_src1 + start_row*input_width);
            pool_height--;

            p_src1_temp = p_src1;
            p_dst_temp = p_dst;
            /* prime */
            align_src1 = AE_LA64_PP(p_src1_temp);
            for(i = 0; i < (input_width >> 2); i++)
            {
                xthalfx4 i1;
                AE_LAHX4IP(i1, align_src1, p_src1_temp);
                AE_SHX4IP(i1, p_dst_temp, 8);
            }
            /* remainder loop for input_width */
            if(input_width & 3)
            {
                xthalf i1h;
                int rem_idx;
                for(rem_idx = 0; rem_idx < (input_width & 3); rem_idx++)
                {
                    i1h = ((xthalf *)p_src1_temp)[rem_idx];
                    ((xthalf *)p_dst_temp)[rem_idx] = i1h;
                }
            }

            p_src2 = p_src1;
            p_src1 = (const xthalfx4 *)p_dst;
            /* Add rows per iteration */
            while(pool_height)
            {
                p_src2 = (const xthalfx4 *)((const xthalf *)p_src2 + input_width);

                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_dst_temp = p_dst;
                /* prime */
                align_src2 = AE_LA64_PP(p_src2_temp);
                for(i = 0; i < (input_width >> 2); i++)
                {
                    xthalfx4 i1, i2, out;

                    AE_LHX4IP(i1, p_src1_temp, 8);
                    AE_LAHX4IP(i2, align_src2, p_src2_temp);

                    out = ADD_HX4(i1, i2);
                    AE_SHX4IP(out, p_dst_temp, 8);
                }
                /* remainder loop for input_width */
                if(input_width & 3)
                {
                    int rem_idx;

                    for(rem_idx = 0; rem_idx < (input_width & 3); rem_idx++)
                    {
                        xthalfx4 i1h, i2h, outh;
                        i1h = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&((xthalf *)p_src1_temp)[rem_idx]));
                        i2h = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&((xthalf *)p_src2_temp)[rem_idx]));
                        outh = ADD_HX4(i1h, i2h);
                        *(WORD16*)&((xthalf *)p_dst_temp)[rem_idx] = AE_MOVAD16_0(AE_MOVINT16X4_FROMXTHALFX4(outh));
                    }
                }
                pool_height--;
            };
        }
        else
        {
            /* If there is no valid input present, fill the output with zero */
            p_dst_pad = p_scratch + left_pad_aligned;
            for(i = 0; i < input_width; i++)
            {
                p_dst_pad[i] = CONST_H(0);
            }
        }

        /* Pool width processing */

        /* On scratch, add width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD16));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (xthalfx4 *)(p_scratch + total_out_width);
        pool_width = kernel_width;

        p_src1 = (const xthalfx4 *)(p_scratch + left_pad_aligned - x_padding);
        pool_width--;

        p_src1_temp = p_src1;
        p_dst_temp = p_dst;
        /* prime */
        align_src1 = AE_LA64_PP(p_src1_temp);
        for(i = 0; i < (scratch_width >> 2); i++)
        {
            xthalfx4 src1;
            AE_LAHX4IP(src1, align_src1, p_src1_temp);
            AE_SHX4IP(src1, p_dst_temp, 8);
        }
        /* remainder loop for scratch_width */
        if(scratch_width & 3)
        {
            xthalf src1h;
            int rem_idx;
            for(rem_idx = 0; rem_idx < (scratch_width & 3); rem_idx++)
            {
                src1h = ((xthalf *)p_src1_temp)[rem_idx];
                ((xthalf *)p_dst_temp)[rem_idx] = src1h;
            }
        }

        p_src2 = p_src1;
        p_src1 = (const xthalfx4 *)p_dst;

        while(pool_width > 0)
        {
            p_src2 = (const xthalfx4 *)((const xthalf *)p_src2 + 1);

            p_src1_temp = p_src1;
            p_src2_temp = p_src2;
            p_dst_temp = p_dst;
            /* prime */
            align_src2 = AE_LA64_PP(p_src2_temp);
            for(i = 0; i < (scratch_width >> 2); i++)
            {
                xthalfx4 src1, src2, out;
                AE_LHX4IP(src1, p_src1_temp, 8);
                AE_LAHX4IP(src2, align_src2, p_src2_temp);

                out = ADD_HX4(src1, src2);
                AE_SHX4IP(out, p_dst_temp, 8);
            }
            /* remainder loop for scratch_width */
            if(scratch_width & 3)
            {
                int rem_idx;

                for(rem_idx = 0; rem_idx < (scratch_width & 3); rem_idx++)
                {
                    xthalfx4 src1h, src2h, outh;
                    src1h = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&((xthalf *)p_src1_temp)[rem_idx]));
                    src2h = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(*(WORD16*)&((xthalf *)p_src2_temp)[rem_idx]));
                    outh = ADD_HX4(src1h, src2h);
                    *(WORD16*)&((xthalf *)p_dst_temp)[rem_idx] = AE_MOVAD16_0(AE_MOVINT16X4_FROMXTHALFX4(outh));
                }
            }
            pool_width--;
        };

        xthalf *ptr_out1 = (xthalf *)(p_scratch + total_out_width);
        xthalf den_inv, den1_inv;
        if(not_last_channel)
        {
            itr_ow = 0;
            den_inv = p_out[itr_oh*out_width+itr_ow];
            den1_inv = p_out[itr_oh*out_width+itr_ow+1];
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_H(ptr_out1[itr_ow*x_stride], den_inv);
                p_out[itr_oh*out_width+itr_ow+1] = MUL_H(ptr_out1[itr_ow*x_stride+x_stride], den1_inv);
                /* store 1/den for next channel */
                p_out[out_plane_size + itr_oh*out_width+itr_ow] = den_inv;
                p_out[out_plane_size + itr_oh*out_width+itr_ow+1] = den1_inv;
                den_inv = p_out[itr_oh*out_width+itr_ow+2];
                den1_inv = p_out[itr_oh*out_width+itr_ow+3];
            }
            if(out_width & 1)
            {
                p_out[itr_oh*out_width+itr_ow]   = MUL_H(ptr_out1[itr_ow*x_stride], den_inv);
                /* store 1/den for next channel */
                p_out[out_plane_size + itr_oh*out_width+itr_ow] = den_inv;

            }
        }
        else
        {
            itr_ow = 0;
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                den_inv = p_out[itr_oh*out_width+itr_ow];
                den1_inv = p_out[itr_oh*out_width+itr_ow+1];
                p_out[itr_oh*out_width+itr_ow]   = MUL_H(ptr_out1[itr_ow*x_stride], den_inv);
                p_out[itr_oh*out_width+itr_ow+1] = MUL_H(ptr_out1[itr_ow*x_stride+x_stride], den1_inv);
            }
            if(out_width & 1)
            {
                den_inv = p_out[itr_oh*out_width+itr_ow];
                p_out[itr_oh*out_width+itr_ow]   = MUL_H(ptr_out1[itr_ow*x_stride], den_inv);

            }
        }
    }
}
#endif /* HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_avgpool_f16,(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_inp,
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
    VOID   *p_scratch))
#else /* #if !HAVE_HP_VFPU */
WORD32 xa_nn_avgpool_f16(
    WORD16* __restrict__ p_out_t,
    const WORD16* __restrict__ p_inp,
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
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out_t, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out_t, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
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
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);

    // Different I/O formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);

    xthalf * __restrict__ p_out = (xthalf*)p_out_t;
    if((input_channels == 1) || (out_data_format == 1))
    {
        // #include <stdio.h>
        // printf("out_data_format == 1\n");
        /* NCHW path: use scratch for intermediate results */
        /* For f16 NCHW, we reuse the same scratch layout as f32 NCHW
         * but with xthalf elements (2 bytes each instead of 4) */
        xa_nn_avgpool_init(-2,
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
        xthalf *p_tmp_out = (xthalf *)(p_state->p_tmp_out);
        int itr_ic, itr_oh, itr_ow;
        const WORD16 *pt_inp;
        WORD16 *pt_out;

        /* Calculate denominators for division */
        
#if 0  /* Original: Compute 1/den in f32, then convert to f16 and store in p_out */
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
                /* Compute denominator in f32 then convert scalar result to f16 */
                xtfloat den_f32 = FLOAT_S(((kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start)), 0);
                xtfloat recip_f32 = MAX_S(RECIP_S(den_f32), ZERO_S());
                /* Store reciprocal as xthalf in output buffer */
                xthalf recip_h = (xthalf)recip_f32;
                p_out[itr_oh*out_width+itr_ow] = recip_h;
            }
        }
#else  /* Compute directly in f16 for testing precision */
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
                /* Compute denominator directly in f16 */
                int area = (kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start);
                #if 0 /* Build fix, the disbaled code builds on HiFi5s but not on HiFi5/HiFi5e RJ3 */
                xthalf den_h = (xthalf)area; 
                xthalf recip_h = RECIP_H(den_h);
                recip_h = MAX_H(recip_h, CONST_H(0));
                p_out[itr_oh*out_width+itr_ow] = recip_h;
                #else
                xtfloatx2 den_f32 = XT_FLOAT_SX2(AE_MOVDA32X2(area, area), 0);
                xtfloatx2 recip_f32_vec = XT_MAX_SX2(XT_RECIP_SX2(den_f32), ZERO_SX2());
                xthalfx4 recip_h = CVTF16S_L(recip_f32_vec);
                ae_int16x4 recip_h_int = AE_MOVINT16X4_FROMXTHALFX4(recip_h);
                recip_h_int = AE_SEL16I(recip_h_int, recip_h_int, 3);
                AE_S16_0_I(recip_h_int, (void *)&p_out[itr_oh*out_width+itr_ow], 0);
                #endif
            }
        }
#endif

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = (WORD16 *)&p_out[itr_ic * out_height * out_width];

            avgpool_f16(pt_out
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
    }
    else
    {
        WORD16 *p_rec_den, *p_den, *p_zeros_mem;
        void *p_scratch_aligned;
        int itr_oh, itr_ow;

        p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        p_rec_den = (WORD16 *)((WORD8 *)p_scratch_aligned +
            2*ALIGNED_SIZE((sizeof(WORD16) * input_channels * input_width), ALIGNMENT));

        p_den = p_rec_den;

        /* Calculate denominators for division */
        xthalf *pxt_rec_den = (xthalf*)p_rec_den;
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

                #if 0 /* Build fix, the disbaled code builds on HiFi5s but not on HiFi5/HiFi5e RJ3 */
                /* Compute denominator in f32 then convert to f16 */
                xtfloat den_f32 = FLOAT_S(((kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start)), 0);
                xtfloat recip_f32 = XT_MAX_S(XT_RECIP_S(den_f32), ZERO_S());
                xthalf recip_h = (xthalf)recip_f32;
                pxt_rec_den[itr_oh*out_width+itr_ow] = recip_h;
                #else
                /* Compute denominator directly in f16 for testing precision */
                WORD32 area = (kernel_y_end-kernel_y_start)*(kernel_x_end-kernel_x_start);
                xtfloatx2 den_f32 = XT_FLOAT_SX2(AE_MOVDA32X2(area, area), 0);
                xtfloatx2 recip_f32 = XT_MAX_SX2(XT_RECIP_SX2(den_f32), ZERO_SX2());
                xthalfx4 recip_h = CVTF16S_L(recip_f32);
                ae_int16x4 recip_h_int = AE_MOVINT16X4_FROMXTHALFX4(recip_h);
                recip_h_int = AE_SEL16I(recip_h_int, recip_h_int, 3);
                AE_S16_0_I(recip_h_int, (void *)&pxt_rec_den[itr_oh*out_width+itr_ow], 0);
                #endif
            }
        }

        p_rec_den = (WORD16 *)((WORD8 *)p_scratch_aligned + ALIGNED_SIZE((sizeof(WORD16) * input_channels * input_width), ALIGNMENT));
        p_zeros_mem = p_rec_den;
        for(itr_oh = 0; itr_oh < input_channels*input_width; itr_oh++)
        {
            p_rec_den[itr_oh] = 0;
        }

        WORD16 * __restrict__ pt_out = (WORD16*)p_out;
        xa_nn_avgpool_f16_hwc(pt_out
                ,p_inp
                ,input_height
                ,input_width
                ,input_channels
                ,kernel_height
                ,kernel_width
                ,x_stride
                ,y_stride
                ,x_padding
                ,y_padding
                ,out_height
                ,out_width
                ,p_scratch_aligned
                ,p_zeros_mem
                ,p_den);
    }
    return 0;
}
#endif /* #if !HAVE_HP_VFPU */
