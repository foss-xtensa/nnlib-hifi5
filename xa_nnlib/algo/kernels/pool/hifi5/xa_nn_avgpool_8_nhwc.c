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
#include "xa_nnlib_common.h"
#include "xa_nn_avgpool_state.h"
#include <math.h>

#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }\
        else\
        {\
            ptr = (WORD8 *)p_zeros_mem;\
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (ptr) + ((n) * (row_size));

#define INCR_ROW_IF_WIDTH_32(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD32 *)p_zeros_mem;\
        }

#define INCR_ROW_IF_WIDTH_16(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD16 *)p_zeros_mem;\
        }

/* Average pooling without using extra copy of input data
 * Works with unaligned input, output.
 */
void xa_nn_avgpool_8_hwc_16(
        WORD8* __restrict__ p_out,
const   WORD8* __restrict__ p_inp,
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
        pVOID    p_zeros_mem,
        WORD32   *p_den_height,
        WORD32   *p_den_width)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD16 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x4 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int16x4 * p_dst, *p_dst_temp;
    ae_int32x2 * p_dst_temp_w, *p_src1_32x2;
    WORD8 *p_out_temp;
    ae_int16x4 * p_src1_scratch;
    ae_valignx2 align_src1, align_src2, align_src3, align_dst;
    int i;
    WORD16 *p_dst_pad;
    ae_int8x8 ZERO8 = AE_MOVDA8(0);
    ae_int16x4 ZERO16 = AE_ZERO16();
    ae_int32x2 ZERO32 = AE_ZERO32();

    plane_size = input_width * input_channels;

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;

        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/

        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int16x4 *)p_scratch ;

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
                    ae_int16x4 wi1, wi2, wj1, wj2;
                    AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                    AE_ADDW8(wi1, wi2, i1, i2);
                    AE_ADDW8(wj1, wj2, j1, j2);
                    AE_ACCW8(wi1, wi2, ZERO8, i3);
                    AE_ACCW8(wj1, wj2, ZERO8, j3);
                    AE_SA16X4X2_IP(wi1, wi2, align_dst, (ae_int16x8 *)p_dst_temp);
                    AE_SA16X4X2_IP(wj1, wj2, align_dst, (ae_int16x8 *)p_dst_temp);
                }

                AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop for input_width */
                for(i = 0; i < (plane_size & 15); i++)
                {
                    ae_int8x8 i1, i2, i3;
                    ae_int16x4 wi1, wi2;
                    AE_L8_IP(i1, (ae_int8 *)p_src1_temp,1);
                    AE_L8_IP(i2, (ae_int8 *)p_src2_temp,1);
                    AE_L8_IP(i3, (ae_int8 *)p_src3_temp,1);

                    AE_ADDW8(wi1, wi2, i1, i2);
                    AE_ACCW8(wi1, wi2, ZERO8, i3);

                    AE_S16_0_IP(wi1, (ae_int16 *)p_dst_temp, 2);
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

                    for(i = 0; i < (plane_size >> 4); i++)
                    {
                        ae_int8x8 i2, i3, j2, j3;
                        ae_int16x4 i0, i1, j0, j1;
                        AE_L16X4X2_IP(i0, i1, (ae_int16x8 *)p_src1_scratch, 16);
                        AE_L16X4X2_IP(j0, j1, (ae_int16x8 *)p_src1_scratch, 16);
                        AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                        AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                        AE_ACCW8(i0, i1, i2, i3);
                        AE_ACCW8(j0, j1, j2, j3);
                        AE_S16X4X2_IP(i0, i1, (ae_int16x8 *)p_dst_temp, 16);
                        AE_S16X4X2_IP(j0, j1, (ae_int16x8 *)p_dst_temp, 16);
                    }

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 15); i++)
                    {
                        ae_int8x8 i2, i3;
                        ae_int16x4 wi1, wi2 = ZERO16;

                        AE_L16_IP(wi1,  (ae_int16 *)p_src1_scratch, 2);
                        AE_L8_IP(i2, (ae_int8 *)p_src2_temp,1);
                        AE_L8_IP(i3, (ae_int8 *)p_src3_temp,1);

                        AE_ACCW8(wi1, wi2, i2, i3);

                        AE_S16_0_IP(wi1, (ae_int16 *)p_dst_temp, 2);
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
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD16 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] =  0; //-INFINITY;
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels);
            p_dst = (ae_int16x4 *)((WORD16 *)p_scratch + ALIGNED_SIZE(plane_size, ALIGNMENT/sizeof(WORD16)));

            if(pool_width)
            {
                p_src1_w = (WORD16 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                // 1st instance
                {
                    p_dst_temp_w = (ae_int32x2 *)p_dst;
                    p_src1_temp_w = (ae_int16x4 *)p_src1_w;
                    p_src2_temp_w = (ae_int16x4 *)p_src2_w;
                    p_src3_temp_w = (ae_int16x4 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA128_PP(p_src1_temp_w);
                    align_src2 = AE_LA128_PP(p_src2_temp_w);
                    align_src3 = AE_LA128_PP(p_src3_temp_w);
                    align_dst = AE_ZALIGN128(); // zero alignment reg

                    for(i = 0; i < (input_channels >> 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, j1, j2, j3;
                        ae_int32x2 wout1, wout2, wout3, wout4;

                        AE_LA16X4X2_IP(i1, j1, align_src1, (ae_int16x8 *)p_src1_temp_w);
                        AE_LA16X4X2_IP(i2, j2, align_src2, (ae_int16x8 *)p_src2_temp_w);
                        AE_LA16X4X2_IP(i3, j3, align_src3, (ae_int16x8 *)p_src3_temp_w);

                        AE_ADDW16(wout1, wout2, ZERO16, i1);
                        AE_ADDW16(wout3, wout4, ZERO16, j1);
                        AE_ACCW16(wout1, wout2, i2, i3);
                        AE_ACCW16(wout3, wout4, j2, j3);

                        AE_SA32X2X2_IP(wout1, wout2, align_dst, (ae_int32x4 *)p_dst_temp_w);
                        AE_SA32X2X2_IP(wout3, wout4, align_dst, (ae_int32x4 *)p_dst_temp_w);
                    }

                    AE_SA128POS_FP(align_dst, p_dst_temp_w); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (input_channels & 7); i++)
                    {
                        ae_int16x4 i1, i2, i3;
                        ae_int32x2 wout1, wout2;

                        AE_L16_IP(i1, (ae_int16 *)p_src1_temp_w,2);
                        AE_L16_IP(i2, (ae_int16 *)p_src2_temp_w,2);
                        AE_L16_IP(i3, (ae_int16 *)p_src3_temp_w,2);

                        AE_ADDW16(wout1, wout2, i1, i2);
                        AE_ACCW16(wout1, wout2, ZERO16, i3);

                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, sizeof(WORD32));
                    }
                }

                if(pool_width)
                {
                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    /* Compare three rows per iteration */
                    do
                    {
                        p_dst_temp_w = (ae_int32x2 *)p_dst;
                        p_src1_32x2 = (ae_int32x2 *)p_dst;
                        p_src2_temp_w = (ae_int16x4 *)p_src2_w;
                        p_src3_temp_w = (ae_int16x4 *)p_src3_w;

                        /* prime */
                        align_src2 = AE_LA128_PP(p_src2_temp_w);
                        align_src3 = AE_LA128_PP(p_src3_temp_w);

                        for(i = 0; i < (input_channels >> 3); i++)
                        {
                            ae_int16x4 i2, i3, j2, j3;
                            ae_int32x2 wout1, wout2, wout3, wout4;

                            AE_L32X2X2_IP(wout1, wout2, (ae_int32x4 *)p_src1_32x2, 16);
                            AE_L32X2X2_IP(wout3, wout4, (ae_int32x4 *)p_src1_32x2, 16);
                            AE_LA16X4X2_IP(i2, j2, align_src2, (ae_int16x8 *)p_src2_temp_w);
                            AE_LA16X4X2_IP(i3, j3, align_src3, (ae_int16x8 *)p_src3_temp_w);

                            AE_ACCW16(wout1, wout2, i2, i3);
                            AE_ACCW16(wout3, wout4, j2, j3);

                            AE_S32X2X2_IP(wout1, wout2, (ae_int32x4 *)p_dst_temp_w, 16);
                            AE_S32X2X2_IP(wout3, wout4, (ae_int32x4 *)p_dst_temp_w, 16);
                        }

                        /* remainder loop */
                        for(i = 0; i < (input_channels & 7); i++)
                        {
                            ae_int16x4 i2, i3;
                            ae_int32x2 wout1, wout2 = ZERO32;

                            AE_L32_IP(wout1, (ae_int32 *)p_src1_32x2,4);
                            AE_L16_IP(i2, (ae_int16 *)p_src2_temp_w,2);
                            AE_L16_IP(i3, (ae_int16 *)p_src3_temp_w,2);

                            AE_ACCW16(wout1, wout2, i2, i3);

                            AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, sizeof(WORD32));
                        }


                        if(!pool_width)
                            break;

                        p_src2_w = p_src3_w;
                        INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                        p_src3_w = p_src2_w;
                        INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    }while(1);
                }

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp32hw;
                ae_int32x2 d0_tmp32, d1_tmp32;
                ae_int32x2 d_out1, d_out2;
                ae_int8x8 d0_out8, d1_out8, d2_out8, d3_out8;
                ae_int64 d_tmp;
                WORD32 *p_out1;

                p_out1 = (WORD32 *)p_dst;

                den_h = AE_MOVDA32(p_den_height[itr_oh]);
                den_w = AE_MOVDA32(p_den_width[itr_ow]);
                d_tmp = AE_MUL32U_LL(den_h, den_w);

                /* Max value of den_h or den_w is 0x80000000
                   so 1 left shift is possible without overflow */
                d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);

                int rem_inp_chan = input_channels - (input_channels%4);

                for(i = 0; i < (input_channels >> 2); i++)
                {
                    AE_L32X2X2_IP(d_out1, d_out2, (ae_int32x4 *)p_out1, 16);
                    
                    d0_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                    d1_tmp32 = AE_MULFP32X2RS(d_out2, d_tmp32hw);
                    
                    d0_out8 = AE_SAT8X4X32_L(d0_tmp32, d0_tmp32);
                    d1_out8 = AE_SEL8X8I(d0_out8, d0_out8, 26);
                    d2_out8 = AE_SAT8X4X32_L(d1_tmp32, d1_tmp32);
                    d3_out8 = AE_SEL8X8I(d2_out8, d2_out8, 26);
                    
                    AE_S8_0_I(d1_out8, (ae_int8 *)&p_out_temp[4*i+0], 0);
                    AE_S8_0_I(d0_out8, (ae_int8 *)&p_out_temp[4*i+1], 0);
                    AE_S8_0_I(d3_out8, (ae_int8 *)&p_out_temp[4*i+2], 0);
                    AE_S8_0_I(d2_out8, (ae_int8 *)&p_out_temp[4*i+3], 0);
                }
                for(i = 0; i < (input_channels & 3); i++)
                {
                    AE_L32_IP(d_out1, (ae_int32 *)p_out1,4);
                    d0_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                    p_out_temp[rem_inp_chan + i] = (WORD8)AE_MOVAD32_L(AE_SRAI32(d0_tmp32, 0));
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (WORD8)0x0;
                }
            }
        }
    }
}

void xa_nn_avgpool_8_hwc_32(
        WORD8* __restrict__ p_out,
const   WORD8* __restrict__ p_inp,
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
        pVOID    p_zeros_mem,
        WORD32   *p_den_height,
        WORD32   *p_den_width)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    int i;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int32x2 * p_src1_scratch;
    WORD32 * p_src1_w, * p_src2_w, * p_src3_w;
    ae_int32x2 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int32x2 * p_dst, *p_dst_temp;
    WORD8 *p_out_temp;
    ae_valignx2 align_src1, align_src2, align_src3, align_dst;
    WORD32 *p_dst_pad;
    ae_int8x8 ZERO8 = AE_MOVDA8(0);
    ae_int32x2 ZERO32 = AE_MOVDA32(0);
    ae_int16x4 one = AE_MOVDA16(1);

    plane_size = input_width * input_channels;
    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;


        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int32x2 *)p_scratch ;

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
            /* 1st instance: Add three rows per iteration */
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
                    ae_int16x4 wi1, wi2, wj1, wj2;
                    ae_int32x2 wout1, wout2, wout3, wout4;
                    ae_int32x2 wout5, wout6, wout7, wout8;
                    wout1 = wout2 = wout3 = wout4 = ZERO32;
                    wout5 = wout6 = wout7 = wout8 = ZERO32;

                    AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                    AE_ADDW8(wi1, wi2, i1, i2);
                    AE_ADDW8(wj1, wj2, j1, j2);
                    AE_ACCW8(wi1, wi2, ZERO8, i3);
                    AE_ACCW8(wj1, wj2, ZERO8, j3);
                    
                    //ae_int32x2 AE_SEXT32X2D16_10(ae_int16x4 d0);
                    AE_MULA16X4(wout1, wout2, wi1, one);
                    AE_MULA16X4(wout3, wout4, wi2, one);
                    AE_MULA16X4(wout5, wout6, wj1, one);
                    AE_MULA16X4(wout7, wout8, wj2, one);

                    AE_SA32X2X2_IP(wout1, wout2, align_dst,(ae_int32x4 *) p_dst_temp);
                    AE_SA32X2X2_IP(wout3, wout4, align_dst,(ae_int32x4 *) p_dst_temp);
                    AE_SA32X2X2_IP(wout5, wout6, align_dst,(ae_int32x4 *) p_dst_temp);
                    AE_SA32X2X2_IP(wout7, wout8, align_dst,(ae_int32x4 *) p_dst_temp);
                }

                AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                /* remainder loop */
                for(i = 0; i < (plane_size & 15); i++)
                {
                    ae_int16x4 i1, i2, i3;
                    ae_int32x2 wout1, wout2;
                    ae_int16x4 one = AE_MOVDA16(1);

                    i1 = AE_MOVDA16(((WORD8 *)p_src1_temp)[i] );
                    i2 = AE_MOVDA16(((WORD8 *)p_src2_temp)[i] );
                    i3 = AE_MOVDA16(((WORD8 *)p_src3_temp)[i] );

                    AE_MUL16X4 (wout1, wout2, i1, one);
                    AE_MULA16X4(wout1, wout2, i2, one);
                    AE_MULA16X4(wout1, wout2, i3, one);

                    AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, sizeof(WORD32));
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

                    align_dst = AE_ZALIGN128(); // zero alignment reg
                    align_src1 = AE_LA128_PP(p_src1_scratch);

                    align_src2 = AE_LA128_PP(p_src2_temp);
                    align_src3 = AE_LA128_PP(p_src3_temp);

                    for(i = 0; i < (plane_size >> 4); i++)
                    {
                        ae_int8x8 i2, i3, j2, j3;
                        ae_int16x4 wi1, wi2, wj1, wj2;
                        ae_int32x2 wout1, wout2, wout3, wout4;
                        ae_int32x2 wout5, wout6, wout7, wout8;

                        AE_LA32X2X2_IP(wout1, wout2, align_src1,(ae_int32x4 *) p_src1_scratch);
                        AE_LA32X2X2_IP(wout3, wout4, align_src1,(ae_int32x4 *) p_src1_scratch);
                        AE_LA32X2X2_IP(wout5, wout6, align_src1,(ae_int32x4 *) p_src1_scratch);
                        AE_LA32X2X2_IP(wout7, wout8, align_src1,(ae_int32x4 *) p_src1_scratch);

                        AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                        AE_LA8X8X2_IP(i3, j3, align_src3, (ae_int8x16 *)p_src3_temp);
                        
                        AE_ADDW8(wi1, wi2, i2, i3);
                        AE_ADDW8(wj1, wj2, j2, j3);
                        
                        AE_MULA16X4(wout1, wout2, wi1, one);
                        AE_MULA16X4(wout3, wout4, wi2, one);
                        AE_MULA16X4(wout5, wout6, wj1, one);
                        AE_MULA16X4(wout7, wout8, wj2, one);

                        AE_SA32X2X2_IP(wout1, wout2, align_dst,(ae_int32x4 *) p_dst_temp);
                        AE_SA32X2X2_IP(wout3, wout4, align_dst,(ae_int32x4 *) p_dst_temp);
                        AE_SA32X2X2_IP(wout5, wout6, align_dst,(ae_int32x4 *) p_dst_temp);
                        AE_SA32X2X2_IP(wout7, wout8, align_dst,(ae_int32x4 *) p_dst_temp);
                    }

                    AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (plane_size & 15); i++)
                    {
                        ae_int16x4 i2, i3;
                        ae_int32x2 wout1, wout2;
                        ae_int16x4 one = AE_MOVDA16(1);
                        WORD32 *p_w = (WORD32 *)p_src1_scratch;

                        wout1 = AE_MOVDA32(p_w[i]);
                        wout2 = wout1;

                        i2 = AE_MOVDA16(((WORD8 *)p_src2_temp)[i] );
                        i3 = AE_MOVDA16(((WORD8 *)p_src3_temp)[i] );

                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);

                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, sizeof(WORD32));
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
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD32 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] = 0;
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels);
            p_dst = (ae_int32x2 *)((WORD32 *)p_scratch + plane_size);

            if(pool_width)
            {
                p_src1_w = (WORD32 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                /* Add three rows per iteration */
                do
                {
                    p_dst_temp = p_dst;
                    p_src1_temp_w = (ae_int32x2 *)p_src1_w;
                    p_src2_temp_w = (ae_int32x2 *)p_src2_w;
                    p_src3_temp_w = (ae_int32x2 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA128_PP(p_src1_temp_w);
                    align_src2 = AE_LA128_PP(p_src2_temp_w);
                    align_src3 = AE_LA128_PP(p_src3_temp_w);
                    align_dst = AE_ZALIGN128(); // zero alignment reg

                    for(i = 0; i < (input_channels >> 2); i++)
                    {
                        ae_int32x2 i1, i2, i3, j1, j2, j3, out, out1;

                        AE_LA32X2X2_IP(i1, j1, align_src1,(ae_int32x4 *) p_src1_temp_w);
                        AE_LA32X2X2_IP(i2, j2, align_src2,(ae_int32x4 *) p_src2_temp_w);
                        AE_LA32X2X2_IP(i3, j3, align_src3,(ae_int32x4 *) p_src3_temp_w);

                        out = AE_ADD32S(i1, i2);
                        out = AE_ADD32S(out, i3);
                        out1 = AE_ADD32S(j1, j2);
                        out1 = AE_ADD32S(out1, j3);

                        AE_SA32X2X2_IP(out, out1, align_dst, (ae_int32x4 *)p_dst_temp);
                    }

                    AE_SA128POS_FP(align_dst, p_dst_temp); // finalize the stream

                    /* remainder loop */
                    for(i = 0; i < (input_channels & 3); i++)
                    {
                        ae_int32x2 i1, i2, i3, out;

                        i1 = AE_MOVDA32(((WORD32 *)p_src2_temp_w)[i]);
                        i2 = AE_MOVDA32(((WORD32 *)p_src2_temp_w)[i]);
                        i3 = AE_MOVDA32(((WORD32 *)p_src3_temp_w)[i]);

                        out = AE_ADD32S(i2, i2);
                        out = AE_ADD32S(out, i3);

                        AE_S32_L_IP(out, (ae_int32 *)p_dst_temp, sizeof(WORD32));
                    }


                    if(!pool_width)
                        break;

                    p_src1_w = (WORD32 *)p_dst;

                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                }while(1);

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp32, d_out1, d_tmp32hw;
                ae_int64 d_tmp;
                WORD32 *p_out1;

                p_out1 = (WORD32 *)p_dst;

                den_h = AE_MOVDA32(p_den_height[itr_oh]);
                den_w = AE_MOVDA32(p_den_width[itr_ow]);
                d_tmp = AE_MUL32U_LL(den_h, den_w);

                /* Max value of den_h or den_w is 0x80000000
                   so 1 left shift is possible without overflow */

                d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);

                for(i=0; i<input_channels; i++)
                {
                    d_out1 = AE_MOVDA32(p_out1[i]);
                    d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                    p_out_temp[i] = (WORD8)AE_MOVAD32_L(AE_SRAI32(d_tmp32, 0));
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (WORD8)0x0;
                }
            }
        }
    }
}


